"""Durable canonical-source and remote-version tracking for paper ingestion."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

_ARXIV_ID = re.compile(
    r"^(?:arxiv:)?(?P<canonical>(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7}))(?P<version>v\d+)?$",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceVersionIdentity:
    """Canonical source identity plus one observable remote revision."""

    source_id: str
    canonical_source_id: str
    source_type: str
    version_key: str
    paper_version: str = ""
    canonical_arxiv_id: str = ""
    remote_updated_at: str = ""
    remote_revision: str = ""
    pdf_url: str = ""


def _text(metadata: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def resolve_source_version(
    source_id: str,
    metadata: Mapping[str, Any],
) -> SourceVersionIdentity:
    """Resolve arXiv canonical identity independently from its remote version."""

    normalized_source_id = source_id.strip()
    raw_arxiv_id = _text(metadata, "arxiv_id")
    candidates = (raw_arxiv_id, normalized_source_id)
    arxiv_matches = [
        match for candidate in candidates if candidate if (match := _ARXIV_ID.fullmatch(candidate)) is not None
    ]
    arxiv_match = next((match for match in arxiv_matches if match.group("version")), None)
    if arxiv_match is None and arxiv_matches:
        arxiv_match = arxiv_matches[0]
    explicit_version = _text(metadata, "paper_version", "arxiv_version")
    remote_updated_at = _text(metadata, "updated_at", "updated")
    source_type = _text(metadata, "source_type", "source_family", "source") or "academic"
    canonical_arxiv_id = ""
    if arxiv_match is not None:
        canonical_arxiv_id = arxiv_match.group("canonical")
        canonical_source_id = f"arXiv:{canonical_arxiv_id}"
        suffix_version = arxiv_match.group("version") or ""
        paper_version = explicit_version or suffix_version
    else:
        canonical_source_id = normalized_source_id
        paper_version = explicit_version

    if paper_version:
        version_key = paper_version.casefold()
    elif remote_updated_at:
        version_key = f"updated:{remote_updated_at}"
    else:
        version_key = "unversioned"

    revision_payload = {
        "canonical_source_id": canonical_source_id,
        "paper_version": paper_version.casefold(),
        "remote_updated_at": remote_updated_at,
        "version_key": version_key,
    }
    remote_revision = hashlib.sha256(
        json.dumps(revision_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SourceVersionIdentity(
        source_id=normalized_source_id,
        canonical_source_id=canonical_source_id,
        source_type=source_type,
        version_key=version_key,
        paper_version=paper_version,
        canonical_arxiv_id=canonical_arxiv_id,
        remote_updated_at=remote_updated_at,
        remote_revision=remote_revision,
        pdf_url=_text(metadata, "pdf_url"),
    )


class JsonSourceRegistry:
    """Atomic JSON registry shared by all embedding-model collections."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"registry_version": 1, "sources": {}}

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid source registry %s: %s", self.path, exc)
            return self._empty()
        if not isinstance(payload, dict) or not isinstance(payload.get("sources"), dict):
            logger.warning("Ignoring source registry %s because its schema is invalid.", self.path)
            return self._empty()
        return payload

    @staticmethod
    def _version_order(version: Mapping[str, Any]) -> tuple[int, int, str]:
        paper_version = str(version.get("paper_version") or "")
        match = re.fullmatch(r"v(\d+)", paper_version, re.IGNORECASE)
        if match:
            return (2, int(match.group(1)), str(version.get("remote_updated_at") or ""))
        updated_at = str(version.get("remote_updated_at") or "")
        return (1 if updated_at else 0, 0, updated_at)

    def _write(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def snapshot(self) -> dict[str, Any]:
        """Return a detached registry snapshot for diagnostics and tests."""

        with self._lock:
            return json.loads(json.dumps(self._read()))

    def get_version(self, identity: SourceVersionIdentity) -> dict[str, Any] | None:
        with self._lock:
            source = self._read()["sources"].get(identity.canonical_source_id)
            if not isinstance(source, dict):
                return None
            versions = source.get("versions")
            if not isinstance(versions, dict):
                return None
            version = versions.get(identity.version_key)
            return dict(version) if isinstance(version, dict) else None

    def is_older_than_latest(self, identity: SourceVersionIdentity) -> bool:
        """Return whether a newer observed revision already owns this source."""

        with self._lock:
            source = self._read()["sources"].get(identity.canonical_source_id)
            if not isinstance(source, dict):
                return False
            versions = source.get("versions")
            if not isinstance(versions, dict):
                return False
            latest = versions.get(source.get("latest_version_key"))
            return isinstance(latest, dict) and self._version_order(asdict(identity)) < self._version_order(latest)

    def mark_older_versions_superseded(
        self,
        identity: SourceVersionIdentity,
        collection_name: str,
    ) -> tuple[str, ...]:
        """Mark older registry versions inactive after a newer index commits."""

        superseded_source_ids: list[str] = []
        with self._lock:
            payload = self._read()
            source = payload["sources"].get(identity.canonical_source_id)
            if not isinstance(source, dict):
                return ()
            versions = source.get("versions")
            if not isinstance(versions, dict):
                return ()
            current_order = self._version_order(asdict(identity))
            for version_key, version in versions.items():
                if (
                    version_key == identity.version_key
                    or not isinstance(version, dict)
                    or self._version_order(version) >= current_order
                ):
                    continue
                indexes = version.get("indexes")
                indexes = dict(indexes) if isinstance(indexes, dict) else {}
                index = indexes.get(collection_name)
                index = dict(index) if isinstance(index, dict) else {}
                index.update(
                    {
                        "status": "SUPERSEDED",
                        "superseded_by": identity.version_key,
                    }
                )
                indexes[collection_name] = index
                version["indexes"] = indexes
                source_id = str(version.get("source_id") or "")
                if source_id and source_id != identity.source_id:
                    superseded_source_ids.append(source_id)
            self._write(payload)
        return tuple(dict.fromkeys(superseded_source_ids))

    def update_version(
        self,
        identity: SourceVersionIdentity,
        **changes: Any,
    ) -> dict[str, Any]:
        """Merge durable acquisition, artifact, or index state for one version."""

        with self._lock:
            payload = self._read()
            sources = payload["sources"]
            source = sources.get(identity.canonical_source_id)
            if not isinstance(source, dict):
                source = {
                    "canonical_source_id": identity.canonical_source_id,
                    "source_type": identity.source_type,
                    "versions": {},
                }
            versions = source.get("versions")
            if not isinstance(versions, dict):
                versions = {}
            current = versions.get(identity.version_key)
            if not isinstance(current, dict):
                current = {}
            current.update(asdict(identity))
            for key, value in changes.items():
                if key in {"artifacts", "indexes"} and isinstance(value, Mapping):
                    existing = current.get(key)
                    merged = dict(existing) if isinstance(existing, dict) else {}
                    merged.update(value)
                    current[key] = merged
                else:
                    current[key] = value
            versions[identity.version_key] = current
            source["versions"] = versions
            latest_key = str(source.get("latest_version_key") or "")
            latest = versions.get(latest_key)
            if not isinstance(latest, dict) or self._version_order(current) >= self._version_order(latest):
                source["latest_version_key"] = identity.version_key
            source["source_type"] = identity.source_type
            sources[identity.canonical_source_id] = source
            self._write(payload)
            return dict(current)
