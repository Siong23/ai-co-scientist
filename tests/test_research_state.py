import json

import pytest

from app.models import ClaimAssessment, ContextMemory, Hypothesis, ReflectionReport, ResearchGoal
from app.research_state import (
    RESEARCH_STATE_SCHEMA_VERSION,
    LocalJSONResearchStateStore,
    ResearchStateCompatibilityError,
    ResearchStateIdentifierError,
    ResearchStateIntegrityError,
    ResearchStateNotFoundError,
    ResearchStateSchemaError,
    ResearchStateStore,
)
from app.run_store import load_run, save_run


def _populated_session():
    research_id = "research-phase-e"
    goal = ResearchGoal(
        "Determine whether selective inhibition improves recovery.",
        preferences="Prefer causal evidence and explicit controls.",
        idea_attributes="causal, falsifiable, feasible",
        constraints={"organism": "mouse"},
        llm_model="test-model",
        query_rewrite_model="test-query-model",
        num_hypotheses=2,
        generation_temperature=0.31,
        reflection_temperature=0.22,
        elo_k_factor=24,
        top_k_hypotheses=1,
        research_type="causal",
        research_id=research_id,
    )
    context = ContextMemory(research_id=research_id, research_type="causal")
    context.iteration_number = 4
    context.research_plan = {
        "research_type": "causal",
        "primary_hypothesis": "Selective inhibition improves recovery.",
        "hypothesis_pipeline_enabled": True,
    }
    context.sub_questions = ["Which pathway mediates recovery?"]
    context.evidence_requirements = [{"aspect_id": "mechanism", "description": "Direct pathway perturbation evidence"}]

    parent = Hypothesis("H-parent", "Parent", "Selective inhibition improves recovery.")
    parent.elo_score = 1275.5
    parent.novelty_review = "HIGH"
    parent.feasibility_review = "MEDIUM"
    parent.review_comments = ["Add a negative control."]
    parent.review_reference_ids = ["source-1"]
    parent.evidence_source_ids = ["source-1"]
    parent.evidence_refs = ["chunk-results-1"]
    parent.evidence_sources = [
        {
            "source_id": "source-1",
            "title": "Perturbation study",
            "abstract": "SOURCE-ABSTRACT-MUST-NOT-PERSIST",
            "content": "SOURCE-CONTENT-MUST-NOT-PERSIST",
            "embedding": [0.1, 0.2],
            "evidence_refs": [
                {
                    "source_id": "source-1",
                    "chunk_id": "chunk-results-1",
                    "section": "Results",
                    "page": 7,
                    "evidence_type": "full_text",
                    "text": "CHUNK-BODY-MUST-NOT-PERSIST",
                }
            ],
        }
    ]
    parent.references = [
        {
            "source_id": "source-1",
            "chunk_id": "chunk-results-1",
            "doi": "10.1000/example",
            "summary": "REFERENCE-BODY-MUST-NOT-PERSIST",
        }
    ]
    parent.audit_score = 8.4
    parent.audit_verdict = "accept"
    parent.audit_report = {
        "claim_assessments": [
            {
                "claim_id": "claim-mechanism",
                "claim": "Selective inhibition activates the recovery pathway.",
                "support_status": "entailed",
                "source_id": "source-1",
                "chunk_ids": ["chunk-results-1"],
                "evidence_spans": ["EVIDENCE-SPAN-MUST-NOT-PERSIST"],
                "section": "Results",
                "page": 7,
                "evidence_type": "full_text",
                "reason": "The intervention and outcome match.",
            }
        ],
        "scores": {"evidence_validity": 8.4},
    }
    parent.reflection_report = ReflectionReport(
        alignment_score=8,
        novelty_score=7,
        feasibility_score=6,
        plausibility_score=8,
        testability_score=9,
        evidence_quality_score=8,
        expected_research_value_score=7,
        strengths=["Direct perturbation design"],
        weaknesses=["Single organism"],
        recommendation="ACCEPT",
        claims=[
            ClaimAssessment(
                claim="Selective inhibition activates the recovery pathway.",
                status="SUPPORTED",
                confidence=8,
                supporting_evidence=[
                    {
                        "source_id": "source-1",
                        "chunk_id": "chunk-results-1",
                        "text": "REFLECTION-EVIDENCE-BODY-MUST-NOT-PERSIST",
                    }
                ],
            )
        ],
        proposed_tests=["Use a pathway-specific rescue."],
        overall_confidence=8,
    )

    child = Hypothesis("H-child", "Child", "A refined pathway-specific mechanism.")
    child.parent_ids = ["H-parent"]
    child.evolution_strategy = "specificity"
    child.elo_score = 1310.25
    child.is_active = False
    child.deactivation_reason = "near_duplicate"
    context.add_hypothesis(parent)
    context.add_hypothesis(child)

    context.evidence_relationships = [
        {
            "research_id": research_id,
            "question_id": "Q-mechanism",
            "hypothesis_id": "H-parent",
            "claim_id": "claim-mechanism",
            "chunk_id": "chunk-results-1",
            "source_id": "source-1",
            "relation": "supports",
            "evidence_strength": 8.0,
        }
    ]
    context.tournament_results = [
        {
            "iteration": 4,
            "hypothesis_a": "H-parent",
            "hypothesis_b": "H-child",
            "outcome": "B",
            "confidence": 8,
            "elo_a_after": 1275.5,
            "elo_b_after": 1310.25,
            "reasoning": "The child is more specific.",
        }
    ]
    context.meta_review_feedback = [
        {
            "synthesis_mode": "llm",
            "meta_review_critique": ["Broaden organism coverage."],
            "research_overview": {"suggested_next_steps": ["Add a second model."]},
        }
    ]
    context.last_literature_synthesis = {
        "established_findings": [
            {
                "claim": "The pathway changes after intervention.",
                "source_ids": ["source-1"],
                "evidence_refs": [
                    {
                        "source_id": "source-1",
                        "chunk_id": "chunk-results-1",
                        "section": "Results",
                        "page": 7,
                        "text": "SYNTHESIS-CHUNK-BODY-MUST-NOT-PERSIST",
                    }
                ],
            }
        ],
        "knowledge_gaps": ["Replication in a second organism"],
    }
    context.last_retrieved_sources = [
        {
            "source_id": "source-1",
            "content": "RETRIEVED-SOURCE-BODY-MUST-NOT-PERSIST",
            "abstract": "RETRIEVED-ABSTRACT-MUST-NOT-PERSIST",
            "evidence_refs": [
                {
                    "source_id": "source-1",
                    "chunk_id": "chunk-results-1",
                    "section": "Results",
                    "page": 7,
                    "text": "RETRIEVED-CHUNK-BODY-MUST-NOT-PERSIST",
                }
            ],
        }
    ]
    context.last_generation_diagnostics = {
        "evidence_retrieval": {"status": "completed", "source_count": 1},
        "retrieved_documents": [{"source_id": "source-1", "content": "DIAGNOSTIC-BODY-MUST-NOT-PERSIST"}],
        "coverage": {
            "source_id": "source-1",
            "chunk_id": "chunk-results-1",
            "text": "DIAGNOSTIC-CHUNK-MUST-NOT-PERSIST",
        },
    }
    context.last_evolution_attempts = [{"strategy": "specificity", "parent_ids": ["H-parent"], "status": "accepted"}]
    context.last_hypothesis_audits = [parent.audit_report]
    context.proximity_analysis = {
        "clusters": {"H-parent": 0, "H-child": 0},
        "cluster_members": {0: ["H-parent", "H-child"]},
        "diversity_score": 0.42,
        "embeddings": [[0.1, 0.2]],
    }
    pending = {
        "action": "REFLECT",
        "reasoning": "Review the evolved child.",
        "target_hypothesis_ids": ["H-child"],
        "confidence": 0.9,
    }
    context.supervisor_state = {
        "status": "running",
        "pending_tasks": [pending],
        "elo_snapshots": [
            {
                "iteration": 4,
                "step": 3,
                "ratings": {"H-parent": 1275.5, "H-child": 1310.25},
                "top_elo": 1310.25,
                "comparison_count": 1,
            }
        ],
        "last_finalization": {"ready": False, "reason": "Reflection pending"},
    }
    return goal, context, pending


def test_save_and_load_reconstructs_complete_research_state(tmp_path):
    goal, context, pending = _populated_session()
    store = LocalJSONResearchStateStore(root_dir=tmp_path / "state")

    path = store.save(goal, context)
    resumed = store.load(goal.research_id)

    assert isinstance(store, ResearchStateStore)
    assert path == tmp_path / "state" / "research-phase-e.json"
    assert store.exists("research-phase-e")
    assert resumed.research_id == goal.research_id
    assert resumed.schema_version == RESEARCH_STATE_SCHEMA_VERSION
    assert resumed.source_path == path
    assert isinstance(resumed.research_goal, ResearchGoal)
    assert isinstance(resumed.context, ContextMemory)
    assert resumed.research_goal.description == goal.description
    assert resumed.research_goal.constraints == {"organism": "mouse"}
    assert resumed.research_goal.llm_model == "test-model"
    assert resumed.research_goal.research_type == "causal"
    assert resumed.context.iteration_number == 4
    assert resumed.context.research_plan == context.research_plan
    assert resumed.context.sub_questions == context.sub_questions
    assert resumed.context.evidence_requirements == context.evidence_requirements
    assert resumed.context.meta_review_feedback == context.meta_review_feedback
    assert resumed.context.last_evolution_attempts == context.last_evolution_attempts

    restored_parent = resumed.context.hypotheses["H-parent"]
    restored_child = resumed.context.hypotheses["H-child"]
    assert isinstance(restored_parent, Hypothesis)
    assert isinstance(restored_parent.reflection_report, ReflectionReport)
    assert isinstance(restored_parent.reflection_report.claims[0], ClaimAssessment)
    assert restored_parent.reflection_report.recommendation == "ACCEPT"
    assert restored_parent.elo_score == 1275.5
    assert restored_parent.audit_score == 8.4
    assert restored_child.parent_ids == ["H-parent"]
    assert restored_child.evolution_strategy == "specificity"
    assert restored_child.is_active is False
    assert restored_child.deactivation_reason == "near_duplicate"
    assert restored_parent.to_dict()["id"] == "H-parent"

    assert resumed.context.tournament_results == context.tournament_results
    assert resumed.context.proximity_analysis["diversity_score"] == 0.42
    assert "embeddings" not in resumed.context.proximity_analysis
    assert resumed.context.supervisor_state["elo_snapshots"] == context.supervisor_state["elo_snapshots"]
    assert resumed.context.supervisor_state["last_finalization"] == context.supervisor_state["last_finalization"]
    assert resumed.context.supervisor_state["pending_tasks"] == []
    assert resumed.context.supervisor_state["status"] == "suspended"
    assert resumed.suspended_pending_tasks == (pending,)
    assert resumed.context.resume_state["status"] == "resumed"
    assert resumed.context.resume_state["requires_evidence_refresh"] is True
    assert resumed.context.last_retrieved_sources == []
    assert "not restarted automatically" in " ".join(resumed.limitations)


def test_state_retains_exact_claim_chunk_source_relationships_without_evidence_bodies(tmp_path):
    goal, context, _ = _populated_session()
    path = LocalJSONResearchStateStore(root_dir=tmp_path).save(goal, context)
    raw_text = path.read_text(encoding="utf-8")
    document = json.loads(raw_text)
    state = document["research_state"]

    assert document["schema_version"] == 1
    assert document["session_id"] == document["research_id"] == "research-phase-e"
    assert document["integrity"]["algorithm"] == "sha256"
    assert len(document["integrity"]["digest"]) == 64
    assert "last_retrieved_sources" not in state

    exact = {
        (item.get("claim_id"), item.get("chunk_id"), item.get("source_id"), item.get("relation"))
        for item in state["evidence_relationships"]
    }
    assert ("claim-mechanism", "chunk-results-1", "source-1", "supports") in exact
    assert any(
        ref.get("chunk_id") == "chunk-results-1" and ref.get("source_id") == "source-1"
        for ref in state["evidence_references"]
    )

    forbidden_values = (
        "SOURCE-ABSTRACT-MUST-NOT-PERSIST",
        "SOURCE-CONTENT-MUST-NOT-PERSIST",
        "CHUNK-BODY-MUST-NOT-PERSIST",
        "REFERENCE-BODY-MUST-NOT-PERSIST",
        "EVIDENCE-SPAN-MUST-NOT-PERSIST",
        "REFLECTION-EVIDENCE-BODY-MUST-NOT-PERSIST",
        "SYNTHESIS-CHUNK-BODY-MUST-NOT-PERSIST",
        "RETRIEVED-SOURCE-BODY-MUST-NOT-PERSIST",
        "RETRIEVED-ABSTRACT-MUST-NOT-PERSIST",
        "RETRIEVED-CHUNK-BODY-MUST-NOT-PERSIST",
        "DIAGNOSTIC-BODY-MUST-NOT-PERSIST",
        "DIAGNOSTIC-CHUNK-MUST-NOT-PERSIST",
    )
    assert all(value not in raw_text for value in forbidden_values)
    assert '"embedding"' not in raw_text.casefold()
    assert '"embeddings"' not in raw_text.casefold()


def test_default_directory_tracks_runs_environment_after_store_construction(tmp_path, monkeypatch):
    store = LocalJSONResearchStateStore()
    monkeypatch.setenv("CO_SCIENTIST_RUNS_DIR", str(tmp_path))
    goal = ResearchGoal("Environment lookup", research_id="research-environment")
    context = ContextMemory(research_id=goal.research_id)

    path = store.save(goal, context)

    assert path == tmp_path / "research_state" / "research-environment.json"


def test_research_state_is_separate_from_immutable_run_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("CO_SCIENTIST_RUNS_DIR", str(tmp_path))
    goal = ResearchGoal("Keep stores separate", research_id="research-separation")
    context = ContextMemory(research_id=goal.research_id)
    run = save_run(
        research_goal=goal,
        cycle_details={"iteration": 1, "steps": {}},
        status="complete",
        references_html="<p>references</p>",
        results_html="<p>results</p>",
        run_id="run-separation",
    )
    run_path = tmp_path / "runs" / "run-separation.json"
    before = run_path.read_bytes()

    state_path = LocalJSONResearchStateStore().save(goal, context)

    assert state_path.parent == tmp_path / "research_state"
    assert run_path.read_bytes() == before
    assert load_run("run-separation") == run
    assert "run_id" not in json.loads(state_path.read_text(encoding="utf-8"))
    assert "research_state" not in json.loads(run_path.read_text(encoding="utf-8"))


def test_incompatible_schema_fails_before_state_is_reconstructed(tmp_path):
    goal = ResearchGoal("Schema check", research_id="research-schema")
    context = ContextMemory(research_id=goal.research_id)
    store = LocalJSONResearchStateStore(root_dir=tmp_path)
    path = store.save(goal, context)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = RESEARCH_STATE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ResearchStateCompatibilityError, match="incompatible"):
        store.load(goal.research_id)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("research_type", "unknown_mode", "Unsupported saved research type"),
        ("research_state.research_type", "exploratory", "Context research type"),
        ("research_state.research_plan.research_type", "comparative", "Research-plan type"),
        ("research_goal.resolved_research_type", "literature_review", "Resolved research type"),
    ],
)
def test_corrupt_or_inconsistent_research_modes_fail_safely(tmp_path, field, value, message):
    goal = ResearchGoal("Mode check", research_type="causal", research_id="research-mode")
    context = ContextMemory(research_id=goal.research_id, research_type="causal")
    context.research_plan = {"research_type": "causal"}
    store = LocalJSONResearchStateStore(root_dir=tmp_path)
    path = store.save(goal, context)
    document = json.loads(path.read_text(encoding="utf-8"))
    target = document
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ResearchStateSchemaError, match=message):
        store.load(goal.research_id)


def test_integrity_tampering_fails_safely(tmp_path):
    goal = ResearchGoal("Integrity check", research_id="research-integrity")
    context = ContextMemory(research_id=goal.research_id)
    store = LocalJSONResearchStateStore(root_dir=tmp_path)
    path = store.save(goal, context)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["research_goal"]["description"] = "tampered"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ResearchStateIntegrityError, match="verification failed"):
        store.load(goal.research_id)


def test_missing_and_malformed_state_fail_with_typed_errors(tmp_path):
    store = LocalJSONResearchStateStore(root_dir=tmp_path)
    with pytest.raises(ResearchStateNotFoundError):
        store.load("research-missing")

    path = tmp_path / "research-malformed.json"
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ResearchStateSchemaError, match="not valid JSON"):
        store.load("research-malformed")


@pytest.mark.parametrize("research_id", ["../escape", "nested/path", "C:drive", "", "contains spaces"])
def test_unsafe_research_ids_cannot_escape_store(tmp_path, research_id):
    store = LocalJSONResearchStateStore(root_dir=tmp_path)
    with pytest.raises(ResearchStateIdentifierError):
        store.exists(research_id)


def test_repeated_save_preserves_creation_time_and_leaves_no_temporary_files(tmp_path):
    goal = ResearchGoal("Atomic update", research_id="research-atomic")
    context = ContextMemory(research_id=goal.research_id)
    store = LocalJSONResearchStateStore(root_dir=tmp_path)
    path = store.save(goal, context)
    first = json.loads(path.read_text(encoding="utf-8"))
    context.iteration_number = 2

    second_path = store.save(goal, context)
    second = json.loads(second_path.read_text(encoding="utf-8"))

    assert second_path == path
    assert second["created_at"] == first["created_at"]
    assert second["research_state"]["iteration_number"] == 2
    assert second["updated_at"] >= first["updated_at"]
    assert not list(tmp_path.glob("*.tmp"))


def test_goal_and_context_ids_must_match(tmp_path):
    goal = ResearchGoal("ID mismatch", research_id="research-goal")
    context = ContextMemory(research_id="research-context")

    with pytest.raises(ResearchStateIdentifierError, match="does not match"):
        LocalJSONResearchStateStore(root_dir=tmp_path).save(goal, context)
