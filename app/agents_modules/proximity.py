"""Hypothesis proximity and semantic topology analysis agent.

The Proximity Agent asynchronously computes a proximity graph for generated
hypotheses, enabling clustering of similar ideas, de-duplication, and efficient
exploration of the hypothesis landscape.  It mirrors the description in the
Co-Scientist paper (Gottweis et al., Nature 2026).

Key capabilities
----------------
* Pairwise TF-IDF cosine similarity with MD5-keyed caching (fast, offline)
* LLM-confirmed near-duplicate detection (two-stage gate to reduce false
  positives before deactivating a hypothesis)
* Connected-component clustering with LLM-generated thematic labels
* Outlier detection and cluster exemplar selection (highest Elo per cluster)
* Structured result dict consumed by the Supervisor and Meta-review agents
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models import ContextMemory, Hypothesis
from ._compat import _legacy

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_CLUSTER_LABEL_SYSTEM = (
    "You are a scientific research analyst. Respond only with valid JSON. "
    "Do not include markdown fences or any text outside the JSON object."
)

_CLUSTER_LABEL_TMPL = """You are summarizing a cluster of related research hypotheses.

Cluster hypotheses:
{hypotheses}

Provide a concise thematic label (3–7 words) that captures the shared scientific
focus of these hypotheses, and a one-sentence explanation.

Respond with valid JSON only:
{{
  "label": "<short thematic label>",
  "explanation": "<one-sentence explanation>"
}}"""

_NEAR_DUP_SYSTEM = (
    "You are a rigorous scientific peer reviewer. Respond only with valid JSON. "
    "Do not include markdown fences or any text outside the JSON object."
)

_NEAR_DUP_TMPL = """Compare the following two research hypotheses and determine whether they
are substantively near-duplicates (that is, they propose the same core scientific
idea with only superficial differences in wording or framing).

Hypothesis A ({id_a}):
Title: {title_a}
{text_a}

Hypothesis B ({id_b}):
Title: {title_b}
{text_b}

Respond with valid JSON only:
{{
  "near_duplicate": true | false,
  "confidence": <0.0–1.0>,
  "reasoning": "<one sentence>"
}}"""


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _strip_fenced_json(text: str) -> str:
    cleaned = text.strip()
    for fence in ("```json", "```JSON", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence) :]
            break
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _safe_parse_json(raw: str) -> Optional[dict]:
    try:
        return json.loads(_strip_fenced_json(raw))
    except (json.JSONDecodeError, ValueError):
        # Attempt to extract the first {...} block as a fallback
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# LLM-backed helpers (thin wrappers; can be independently mocked in tests)
# ---------------------------------------------------------------------------


def call_llm_for_cluster_label(
    hypothesis_snippets: List[str],
    max_tokens: int = 256,
) -> Dict[str, str]:
    """Ask the LLM for a thematic label for a cluster of hypotheses.

    Returns a dict with keys ``label`` and ``explanation``.  Falls back to a
    generic label on any error so the pipeline is never blocked.
    """
    joined = "\n\n".join(
        f"[{i + 1}] {snip}" for i, snip in enumerate(hypothesis_snippets[:5])
    )
    prompt = _CLUSTER_LABEL_TMPL.format(hypotheses=joined)
    try:
        raw = _legacy.call_llm(
            prompt,
            temperature=0.3,
            system_prompt=_CLUSTER_LABEL_SYSTEM,
            max_tokens=max_tokens,
        )
        parsed = _safe_parse_json(raw)
        if parsed and "label" in parsed:
            return {
                "label": str(parsed["label"]).strip(),
                "explanation": str(parsed.get("explanation", "")).strip(),
            }
    except Exception as exc:  # pragma: no cover
        _legacy.logger.warning("Cluster label LLM call failed: %s", exc)

    return {"label": "Mixed Research Directions", "explanation": ""}


def call_llm_for_near_duplicate_check(
    hypo_a: Hypothesis,
    hypo_b: Hypothesis,
    max_tokens: int = 256,
) -> Dict[str, Any]:
    """Ask the LLM whether two hypotheses are near-duplicates.

    Returns a dict with keys ``near_duplicate`` (bool), ``confidence`` (float),
    and ``reasoning`` (str).  Defaults to ``near_duplicate=False`` on failure.
    """
    text_a = f"{hypo_a.title}\n{hypo_a.text}" if hypo_a.title else hypo_a.text
    text_b = f"{hypo_b.title}\n{hypo_b.text}" if hypo_b.title else hypo_b.text
    # Truncate to keep token usage bounded
    max_chars = 800
    prompt = _NEAR_DUP_TMPL.format(
        id_a=hypo_a.hypothesis_id,
        title_a=hypo_a.title or hypo_a.hypothesis_id,
        text_a=text_a[:max_chars],
        id_b=hypo_b.hypothesis_id,
        title_b=hypo_b.title or hypo_b.hypothesis_id,
        text_b=text_b[:max_chars],
    )
    try:
        raw = _legacy.call_llm(
            prompt,
            temperature=0.1,
            system_prompt=_NEAR_DUP_SYSTEM,
            max_tokens=max_tokens,
        )
        parsed = _safe_parse_json(raw)
        if parsed is not None:
            return {
                "near_duplicate": bool(parsed.get("near_duplicate", False)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
            }
    except Exception as exc:  # pragma: no cover
        _legacy.logger.warning("Near-duplicate LLM call failed: %s", exc)

    return {"near_duplicate": False, "confidence": 0.0, "reasoning": ""}


# ---------------------------------------------------------------------------
# ProximityAgent
# ---------------------------------------------------------------------------


class ProximityAgent:
    """Proximity Agent: Analyzes semantic relationships, clusters, diversity,
    and topology of hypotheses.

    Parameters
    ----------
    similarity_threshold:
        Minimum cosine similarity for an edge to appear in the adjacency graph.
    cluster_threshold:
        Minimum cosine similarity for two hypotheses to be placed in the same
        connected component.
    outlier_threshold:
        Hypotheses whose mean pairwise similarity falls below this value are
        classified as outliers (isolated ideas).
    near_dup_sim_threshold:
        Cosine-similarity gate: only pairs above this value are submitted to the
        LLM for near-duplicate confirmation.  Lower values produce more LLM calls.
    near_dup_llm_confidence:
        Minimum LLM confidence required to classify a pair as near-duplicate.
    label_clusters:
        When *True* the LLM generates a thematic label for every cluster with
        ≥ 2 members.  Set to *False* to skip LLM calls (faster offline testing).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.2,
        cluster_threshold: float = 0.45,
        outlier_threshold: float = 0.25,
        near_dup_sim_threshold: float = 0.70,
        near_dup_llm_confidence: float = 0.75,
        label_clusters: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.cluster_threshold = cluster_threshold
        self.outlier_threshold = outlier_threshold
        self.near_dup_sim_threshold = near_dup_sim_threshold
        self.near_dup_llm_confidence = near_dup_llm_confidence
        self.label_clusters = label_clusters
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _text_hash(self, text: str) -> str:
        """Returns MD5 hex digest of normalized text for caching."""
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    def _get_similarity(self, text_a: str, text_b: str) -> float:
        """Computes similarity with caching."""
        if not text_a or not text_b or not text_a.strip() or not text_b.strip():
            return 0.0

        hash_a = self._text_hash(text_a)
        hash_b = self._text_hash(text_b)
        cache_key = (hash_a, hash_b) if hash_a <= hash_b else (hash_b, hash_a)

        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        try:
            score = float(_legacy.similarity_score(text_a, text_b))
        except Exception as e:
            _legacy.logger.warning(f"Error computing similarity: {e}")
            score = 0.0

        self._similarity_cache[cache_key] = score
        return score

    def _cluster_hypotheses(
        self, hypotheses: List[Hypothesis], sim_matrix: List[List[float]]
    ) -> Dict[int, List[str]]:
        """Groups hypotheses into clusters via BFS connected components."""
        n = len(hypotheses)
        visited: Set[int] = set()
        clusters: Dict[int, List[str]] = {}
        cluster_id = 0

        for i in range(n):
            if i in visited:
                continue
            component: List[str] = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                component.append(hypotheses[curr].hypothesis_id)
                for neighbor in range(n):
                    if (
                        neighbor not in visited
                        and sim_matrix[curr][neighbor] >= self.cluster_threshold
                    ):
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters[cluster_id] = component
            cluster_id += 1

        return clusters

    def _generate_cluster_labels(
        self,
        clusters: Dict[int, List[str]],
        hypo_map: Dict[str, Hypothesis],
    ) -> Dict[int, Dict[str, str]]:
        """Generate thematic labels for clusters with ≥ 2 members via LLM."""
        labels: Dict[int, Dict[str, str]] = {}
        for c_id, member_ids in clusters.items():
            if len(member_ids) < 2:
                # Single-member clusters get a label from the hypothesis title
                h = hypo_map.get(member_ids[0]) if member_ids else None
                labels[c_id] = {
                    "label": (h.title or member_ids[0]) if h else "Single Hypothesis",
                    "explanation": "",
                }
                continue
            snippets = []
            for m_id in member_ids[:5]:
                h = hypo_map.get(m_id)
                if h:
                    snippet = h.title or m_id
                    if h.text:
                        snippet += ": " + (h.text[:150] + "..." if len(h.text) > 150 else h.text)
                    snippets.append(snippet)
            labels[c_id] = call_llm_for_cluster_label(snippets)
        return labels

    def _find_near_duplicates(
        self,
        hypotheses: List[Hypothesis],
        sim_matrix: List[List[float]],
    ) -> List[Dict[str, Any]]:
        """Identify near-duplicate pairs using a two-stage gate.

        Stage 1 (fast): TF-IDF similarity ≥ near_dup_sim_threshold
        Stage 2 (LLM): LLM confirms pair is substantively near-duplicate with
                        confidence ≥ near_dup_llm_confidence.

        Returns a list of dicts, each with keys:
          ``id_a``, ``id_b``, ``similarity``, ``confidence``, ``reasoning``
        """
        n = len(hypotheses)
        near_dups: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_matrix[i][j]
                if sim < self.near_dup_sim_threshold:
                    continue
                # Stage 2: LLM confirmation
                result = call_llm_for_near_duplicate_check(
                    hypotheses[i], hypotheses[j]
                )
                if (
                    result["near_duplicate"]
                    and result["confidence"] >= self.near_dup_llm_confidence
                ):
                    near_dups.append(
                        {
                            "id_a": hypotheses[i].hypothesis_id,
                            "id_b": hypotheses[j].hypothesis_id,
                            "similarity": round(sim, 3),
                            "confidence": round(result["confidence"], 3),
                            "reasoning": result["reasoning"],
                        }
                    )
        return near_dups

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_proximity_graph(
        self,
        context: ContextMemory,
        *,
        run_near_duplicate_check: bool = True,
    ) -> Dict[str, Any]:
        """Builds proximity graph data, clusters, outliers, near-duplicates,
        and diversity analysis.

        Parameters
        ----------
        context:
            The shared context memory holding all active hypotheses.
        run_near_duplicate_check:
            When *False* the LLM-based near-duplicate confirmation is skipped.
            Useful for unit tests or low-compute runs.

        Returns
        -------
        dict
            Keys:
            * ``adjacency_graph`` – {hyp_id: [{other_id, similarity}, …]}
            * ``nodes`` – vis.js-compatible node list
            * ``edges`` – vis.js-compatible edge list
            * ``clusters`` – {cluster_id: [hyp_id, …]}
            * ``cluster_labels`` – {cluster_id: {label, explanation}}
            * ``outliers`` – [hyp_id, …]
            * ``exemplars`` – [hyp_id, …] (highest Elo per cluster)
            * ``near_duplicates`` – [{id_a, id_b, similarity, confidence, reasoning}]
            * ``diversity_score`` – float 0–1
            * ``mean_similarities`` – {hyp_id: float}
        """
        active_hypotheses = context.get_active_hypotheses()
        if not active_hypotheses:
            _legacy.logger.info("No active hypotheses to build proximity graph.")
            return {
                "adjacency_graph": {},
                "nodes": [],
                "edges": [],
                "clusters": {},
                "cluster_labels": {},
                "outliers": [],
                "exemplars": [],
                "near_duplicates": [],
                "diversity_score": 0.0,
                "mean_similarities": {},
            }

        n = len(active_hypotheses)
        adjacency: Dict[str, List[Dict[str, Any]]] = {
            h.hypothesis_id: [] for h in active_hypotheses
        }

        if n == 1:
            h = active_hypotheses[0]
            visjs_data = _legacy.generate_visjs_data(adjacency)
            single_label = {0: {"label": h.title or h.hypothesis_id, "explanation": ""}}
            return {
                "adjacency_graph": adjacency,
                "nodes": visjs_data.get(
                    "nodes", [{"id": h.hypothesis_id, "label": h.hypothesis_id}]
                ),
                "edges": visjs_data.get("edges", []),
                "clusters": {0: [h.hypothesis_id]},
                "cluster_labels": single_label,
                "outliers": [],
                "exemplars": [h.hypothesis_id],
                "near_duplicates": [],
                "diversity_score": 1.0,
                "mean_similarities": {h.hypothesis_id: 0.0},
            }

        # ----------------------------------------------------------------
        # Step 1: Compute pairwise similarity matrix
        # ----------------------------------------------------------------
        sim_matrix = [[0.0] * n for _ in range(n)]
        all_pair_similarities: List[float] = []

        for i in range(n):
            sim_matrix[i][i] = 1.0
            hypo_i = active_hypotheses[i]
            text_i = f"{hypo_i.title}\n{hypo_i.text}" if hypo_i.title else hypo_i.text

            for j in range(i + 1, n):
                hypo_j = active_hypotheses[j]
                text_j = (
                    f"{hypo_j.title}\n{hypo_j.text}" if hypo_j.title else hypo_j.text
                )

                if text_i and text_j:
                    sim = self._get_similarity(text_i, text_j)
                else:
                    _legacy.logger.warning(
                        "Skipping similarity for %s or %s due to empty text.",
                        hypo_i.hypothesis_id,
                        hypo_j.hypothesis_id,
                    )
                    sim = 0.0

                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim
                all_pair_similarities.append(sim)

                if sim >= self.similarity_threshold:
                    adjacency[hypo_i.hypothesis_id].append(
                        {"other_id": hypo_j.hypothesis_id, "similarity": sim}
                    )
                    adjacency[hypo_j.hypothesis_id].append(
                        {"other_id": hypo_i.hypothesis_id, "similarity": sim}
                    )

        # ----------------------------------------------------------------
        # Step 2: Per-hypothesis mean similarity and overall diversity
        # ----------------------------------------------------------------
        mean_similarities: Dict[str, float] = {}
        for i in range(n):
            other_sims = [sim_matrix[i][j] for j in range(n) if i != j]
            mean_similarities[active_hypotheses[i].hypothesis_id] = (
                sum(other_sims) / len(other_sims) if other_sims else 0.0
            )

        avg_pairwise_sim = (
            sum(all_pair_similarities) / len(all_pair_similarities)
            if all_pair_similarities
            else 0.0
        )
        diversity_score = max(0.0, min(1.0, 1.0 - avg_pairwise_sim))

        # ----------------------------------------------------------------
        # Step 3: Outliers
        # ----------------------------------------------------------------
        outliers: List[str] = [
            h_id
            for h_id, mean_sim in mean_similarities.items()
            if mean_sim < self.outlier_threshold
        ]

        # ----------------------------------------------------------------
        # Step 4: Clustering
        # ----------------------------------------------------------------
        clusters = self._cluster_hypotheses(active_hypotheses, sim_matrix)

        # ----------------------------------------------------------------
        # Step 5: Near-duplicate detection (LLM-confirmed)
        # ----------------------------------------------------------------
        near_duplicates: List[Dict[str, Any]] = []
        if run_near_duplicate_check:
            near_duplicates = self._find_near_duplicates(active_hypotheses, sim_matrix)

        # ----------------------------------------------------------------
        # Step 6: Cluster exemplars (highest Elo per cluster)
        # ----------------------------------------------------------------
        exemplars: List[str] = []
        hypo_map = {h.hypothesis_id: h for h in active_hypotheses}
        for _cluster_id, member_ids in clusters.items():
            cluster_hypos = [
                hypo_map[m_id] for m_id in member_ids if m_id in hypo_map
            ]
            if cluster_hypos:
                best = max(
                    cluster_hypos, key=lambda h: getattr(h, "elo_score", 1200.0)
                )
                exemplars.append(best.hypothesis_id)

        # ----------------------------------------------------------------
        # Step 7: LLM cluster labels
        # ----------------------------------------------------------------
        cluster_labels: Dict[int, Dict[str, str]] = {}
        if self.label_clusters:
            cluster_labels = self._generate_cluster_labels(clusters, hypo_map)
        else:
            for c_id, member_ids in clusters.items():
                h = hypo_map.get(member_ids[0]) if member_ids else None
                cluster_labels[c_id] = {
                    "label": (h.title or member_ids[0]) if h else f"Cluster {c_id + 1}",
                    "explanation": "",
                }

        # ----------------------------------------------------------------
        # Step 8: vis.js graph enrichment
        # ----------------------------------------------------------------
        visjs_data = _legacy.generate_visjs_data(adjacency)
        nodes = visjs_data.get("nodes", [])
        edges = visjs_data.get("edges", [])

        id_to_cluster: Dict[str, int] = {}
        for c_id, member_ids in clusters.items():
            for m_id in member_ids:
                id_to_cluster[m_id] = c_id

        for node in nodes:
            node_id = node.get("id")
            hypo = hypo_map.get(node_id)
            if hypo:
                c_id = id_to_cluster.get(node_id, 0)
                cluster_label = cluster_labels.get(c_id, {}).get(
                    "label", f"Cluster {c_id + 1}"
                )
                node["group"] = cluster_label
                elo = getattr(hypo, "elo_score", 1200.0)
                node["value"] = max(10, min(30, int(elo / 50)))
                title_text = hypo.title or node_id
                snippet = (
                    (hypo.text[:150] + "...") if len(hypo.text) > 150 else hypo.text
                )
                is_dup = any(
                    nd["id_a"] == node_id or nd["id_b"] == node_id
                    for nd in near_duplicates
                )
                dup_flag = " ⚠ near-duplicate" if is_dup else ""
                node["title"] = (
                    f"<b>{node_id}: {title_text}</b>"
                    f"<br>Elo: {elo:.1f} | {cluster_label}{dup_flag}"
                    f"<br><br>{snippet}"
                )

        _legacy.logger.info(
            "Built proximity graph: %d nodes, %d edges, %d clusters, "
            "%d outliers, %d near-duplicates, diversity: %.3f",
            len(active_hypotheses),
            len(edges),
            len(clusters),
            len(outliers),
            len(near_duplicates),
            diversity_score,
        )

        return {
            "adjacency_graph": adjacency,
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
            "cluster_labels": cluster_labels,
            "outliers": outliers,
            "exemplars": exemplars,
            "near_duplicates": near_duplicates,
            "diversity_score": round(diversity_score, 3),
            "mean_similarities": {k: round(v, 3) for k, v in mean_similarities.items()},
        }
