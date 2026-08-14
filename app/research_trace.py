"""Safe formatting helpers for live research-process traces."""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Mapping, MutableSequence, Sequence
from urllib.parse import quote, urlparse

from .utils import redact_secrets

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s<>'\"]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s<>'\"]+"),
)

_STATUS_LABELS = {
    "running": "Running",
    "completed": "Completed",
    "warning": "Completed with warnings",
    "error": "Failed",
}


def _redact(value: Any) -> str:
    redacted = redact_secrets(str(value))
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda match: f"{match.group(1)}***REDACTED***",
            redacted,
        )
    return redacted


def _plain_text(value: Any, *, limit: int) -> str:
    text = " ".join(_redact(value).split())
    if len(text) > limit:
        return f"{text[: limit - 3].rstrip()}..."
    return text


def _normalize_source(source: Mapping[str, Any]) -> Dict[str, str] | None:
    source_id = _plain_text(source.get("source_id") or source.get("id") or "", limit=160)
    url = str(source.get("url") or source.get("arxiv_url") or source.get("pdf_url") or "").strip()
    if not url and source_id.startswith("arXiv:"):
        arxiv_id = source_id.removeprefix("arXiv:")
        url = f"https://arxiv.org/abs/{quote(arxiv_id, safe='/.-')}"

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    domain = parsed.netloc.lower().removeprefix("www.")
    title = _plain_text(source.get("title") or source_id or domain, limit=240)
    return {
        "source_id": source_id or url,
        "title": title,
        "url": url,
        "domain": domain,
    }


def _format_duration(seconds: float | None) -> str:
    total_seconds = max(0, int(round(seconds or 0)))
    minutes, seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def normalize_trace_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a bounded, serializable copy of a progress event."""
    details = event.get("details") or []
    if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
        details = [details]

    raw_sources = event.get("sources") or []
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raw_sources = []
    sources = []
    seen_sources = set()
    for source in raw_sources:
        if not isinstance(source, Mapping):
            continue
        normalized_source = _normalize_source(source)
        if not normalized_source:
            continue
        source_key = normalized_source["url"]
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        sources.append(normalized_source)
        if len(sources) >= 200:
            break

    normalized: Dict[str, Any] = {
        "step": _plain_text(event.get("step") or "unknown", limit=80),
        "status": str(event.get("status") or "running").lower(),
        "title": _plain_text(event.get("title") or "Research step", limit=160),
        "summary": _plain_text(event.get("summary") or "", limit=600),
        "details": [_plain_text(item, limit=500) for item in list(details)[:5] if str(item).strip()],
        "sources": sources,
        "source_count": max(int(event.get("source_count") or 0), len(sources)),
    }
    elapsed = event.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed >= 0:
        normalized["elapsed_seconds"] = round(float(elapsed), 2)
    return normalized


def merge_trace_event(
    events: MutableSequence[Dict[str, Any]],
    event: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Upsert an event by step while preserving the workflow order."""
    normalized = normalize_trace_event(event)
    step = normalized["step"]
    for index, existing in enumerate(events):
        if existing.get("step") == step:
            events[index] = normalized
            break
    else:
        events.append(normalized)
    return list(events)


def format_research_trace_html(
    events: Sequence[Mapping[str, Any]] | None,
    *,
    running: bool = False,
    elapsed_seconds: float | None = None,
) -> str:
    """Render a ChatGPT-style activity disclosure and source drawer."""
    trace: List[Dict[str, Any]] = []
    for event in events or []:
        merge_trace_event(trace, event)

    if elapsed_seconds is None:
        elapsed_seconds = sum(float(event.get("elapsed_seconds") or 0) for event in trace)
    duration = _format_duration(elapsed_seconds)
    duration_label = f"Working for {duration}…" if running else f"Worked for {duration}"

    sources: List[Dict[str, str]] = []
    seen_source_urls = set()
    source_count = 0
    for event in trace:
        source_count = max(source_count, int(event.get("source_count") or 0))
        for source in event.get("sources") or []:
            url = source.get("url")
            if not url or url in seen_source_urls:
                continue
            seen_source_urls.add(url)
            sources.append(source)
    source_count = max(source_count, len(sources))

    timeline_items = []
    for event in trace:
        status = event["status"] if event["status"] in _STATUS_LABELS else "running"
        elapsed = event.get("elapsed_seconds")
        elapsed_html = f'<span class="trace-time">{elapsed:.1f}s</span>' if elapsed is not None else ""
        summary_html = f'<p class="trace-summary">{html.escape(event["summary"])}</p>' if event["summary"] else ""
        details_html = ""
        if event["details"]:
            detail_items = "".join(f"<li>{html.escape(detail)}</li>" for detail in event["details"])
            details_html = f'<ul class="trace-details">{detail_items}</ul>'
        timeline_items.append(
            f'<li class="trace-item trace-{status}">'
            '<span class="trace-dot" aria-hidden="true"></span>'
            '<div class="trace-content">'
            '<div class="trace-heading">'
            f"<strong>{html.escape(event['title'])}</strong>"
            f'<span class="trace-status">{_STATUS_LABELS[status]}</span>'
            f"{elapsed_html}"
            "</div>"
            f"{summary_html}{details_html}"
            "</div></li>"
        )

    if not timeline_items:
        timeline_items.append('<li class="trace-empty">The agent workflow will appear here after the cycle starts.</li>')

    source_trigger = ""
    if source_count:
        noun = "source" if source_count == 1 else "sources"
        source_trigger = (
            '<label class="source-trigger" for="research-activity-drawer-toggle" role="button" tabindex="0">'
            '<span class="source-globe">◎</span>'
            f"Searched {source_count} {noun}"
            '<span class="source-arrow">›</span>'
            "</label>"
        )

    source_chips = "".join(
        f'<a class="source-chip" href="{html.escape(source["url"], quote=True)}" '
        'target="_blank" rel="noopener noreferrer">'
        f'<span class="source-chip-icon">↗</span>{html.escape(source["domain"])}</a>'
        for source in sources[:12]
    )
    if len(sources) > 12:
        source_chips += f'<span class="source-chip source-more">+{len(sources) - 12} more</span>'

    source_cards = "".join(
        '<a class="source-card" '
        f'href="{html.escape(source["url"], quote=True)}" target="_blank" rel="noopener noreferrer">'
        f'<span class="source-domain">{html.escape(source["domain"])}</span>'
        f'<strong>{html.escape(source["title"])}</strong>'
        '<span class="source-open">↗</span>'
        "</a>"
        for source in sources
    )
    if not source_cards:
        source_cards = '<p class="trace-empty">No clickable source URLs were recorded for this run.</p>'

    return f"""
    <style>
      .activity-toggle{{position:absolute;opacity:0;pointer-events:none}}
      .research-trace{{margin:8px 0 18px;color:var(--body-text-color,#111);border-radius:8px;margin:20px 0;padding:15px;}}
      .research-trace>summary{{cursor:pointer;display:inline-flex;gap:8px;align-items:center;padding:8px 0;list-style:none;color:#8a8a8a;font-size:1.05em}}
      .research-trace>summary::-webkit-details-marker{{display:none}}
      .research-trace>summary::after{{content:"›";font-size:1.35em;line-height:1;transition:transform .16s ease}}
      .research-trace[open]>summary::after{{transform:rotate(90deg)}}
      .trace-inline{{max-width:900px;padding:8px 0 0}}
      .trace-note{{margin:0 0 18px;color:#5f6368}}
      .trace-list{{list-style:none;margin:0;padding:0}}
      .trace-item{{display:flex;gap:12px;position:relative;padding:7px 0 11px}}
      .trace-item:not(:last-child)::after{{content:"";position:absolute;left:5px;top:22px;bottom:-2px;width:2px;background:#d8dee9}}
      .trace-dot{{width:12px;height:12px;border-radius:50%;margin-top:5px;flex:0 0 auto;background:#94a3b8}}
      .trace-completed .trace-dot{{background:#16a34a}}
      .trace-warning .trace-dot{{background:#d97706}}
      .trace-error .trace-dot{{background:#dc2626}}
      .trace-running .trace-dot{{background:#2563eb;box-shadow:0 0 0 4px rgba(37,99,235,.14)}}
      .trace-content{{min-width:0;flex:1}}
      .trace-heading{{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;color:#0f172a}}
      .trace-status,.trace-time{{color:#64748b;font-size:.82em}}
      .trace-summary{{margin:4px 0 0;color:#334155}}
      .trace-details{{margin:5px 0 0;padding-left:20px;color:#475569}}
      .trace-empty{{padding:4px 0 8px;color:#64748b}}
      .source-trigger{{display:flex;align-items:center;gap:10px;width:max-content;max-width:100%;margin:12px 0 18px;color:#888;cursor:pointer;font-size:1.02em}}
      .source-trigger:hover{{color:#2563eb}}
      .source-globe{{color:#0ea5e9;font-size:1.35em}}
      .source-arrow{{font-size:1.35em}}
      .activity-backdrop{{display:none;position:fixed;inset:0;background:rgba(15,23,42,.22);z-index:9998;cursor:pointer}}
      .activity-drawer{{position:fixed;z-index:9999;top:0;right:0;height:100vh;width:min(480px,94vw);background:var(--body-background-fill,#fff);color:var(--body-text-color,#111);box-shadow:-12px 0 35px rgba(15,23,42,.18);transform:translateX(105%);transition:transform .22s ease;overflow:auto;text-align:left}}
      .activity-toggle:checked~.activity-backdrop{{display:block}}
      .activity-toggle:checked~.activity-drawer{{transform:translateX(0)}}
      .drawer-header{{position:sticky;top:0;z-index:2;display:flex;align-items:center;padding:18px 22px;border-bottom:1px solid #e5e7eb;background:var(--body-background-fill,#fff)}}
      .drawer-header strong{{font-size:1.15em}}
      .drawer-duration{{color:#888;margin-left:8px}}
      .drawer-close{{margin-left:auto;font-size:1.8em;line-height:1;cursor:pointer;padding:4px 8px}}
      .drawer-body{{padding:18px 24px 32px}}
      .drawer-body h3{{margin:0 0 14px}}
      .drawer-timeline{{margin-bottom:28px}}
      .drawer-search{{margin:4px 0 18px;padding-left:24px}}
      .drawer-search-title{{display:flex;gap:8px;align-items:center;margin-bottom:10px}}
      .source-chips{{display:flex;flex-wrap:wrap;gap:8px}}
      .source-chip{{display:inline-flex;gap:6px;align-items:center;padding:7px 11px;border-radius:999px;background:#f3f4f6;color:#4b5563!important;text-decoration:none!important}}
      .source-chip:hover{{background:#e5e7eb}}
      .source-chip-icon{{font-size:.8em}}
      .source-more{{cursor:default}}
      .sources-heading{{margin-top:28px!important}}
      .source-card{{position:relative;display:flex;flex-direction:column;gap:4px;padding:13px 32px 13px 0;border-bottom:1px solid #e5e7eb;color:inherit!important;text-decoration:none!important}}
      .source-card:hover strong{{color:#2563eb}}
      .source-domain{{color:#6b7280;font-size:.86em}}
      .source-open{{position:absolute;right:4px;top:18px;color:#6b7280}}
      @media(max-width:700px){{.activity-drawer{{width:100vw}}.drawer-body{{padding:16px 18px 28px}}}}
    </style>
    <input class="activity-toggle" id="research-activity-drawer-toggle" type="checkbox" aria-hidden="true">
    <details class="research-trace" data-testid="research-trace">
      <summary><span>{html.escape(duration_label)}</span></summary>
      <div class="trace-inline">
        <p class="trace-note">Activity summary based on agent actions and recorded evidence.</p>
        {source_trigger}
        <ol class="trace-list">{"".join(timeline_items)}</ol>
      </div>
    </details>
    <label class="activity-backdrop" for="research-activity-drawer-toggle" aria-label="Close activity"></label>
    <aside class="activity-drawer" aria-label="Research activity">
      <div class="drawer-header">
        <strong>Activity</strong><span class="drawer-duration">· {html.escape(duration)}</span>
        <label class="drawer-close" for="research-activity-drawer-toggle" aria-label="Close activity">×</label>
      </div>
      <div class="drawer-body">
        <h3>Thinking</h3>
        <div class="drawer-search">
          <div class="drawer-search-title"><span>◎</span><strong>Searched {source_count} sources</strong></div>
          <div class="source-chips">{source_chips}</div>
        </div>
        <ol class="trace-list drawer-timeline">{"".join(timeline_items)}</ol>
        <h3 class="sources-heading">Sources · {source_count}</h3>
        <div class="source-cards">{source_cards}</div>
      </div>
    </aside>
    """.strip()
