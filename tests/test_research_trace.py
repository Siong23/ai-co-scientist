from app.research_trace import format_research_trace_html, merge_trace_event


def test_trace_events_replace_running_stage_and_escape_content(monkeypatch):
    secret = "LMSTUDIO-TRACE-SECRET"
    monkeypatch.setenv("LMSTUDIO_API_KEY", secret)
    events = []

    merge_trace_event(
        events,
        {
            "step": "generation",
            "status": "running",
            "title": "Discovering <evidence>",
            "summary": "Starting the search.",
            "details": [],
        },
    )
    merge_trace_event(
        events,
        {
            "step": "generation",
            "status": "completed",
            "title": "Discovering <evidence>",
            "summary": f"Finished with Authorization: Bearer {secret}",
            "details": ["<script>alert('trace')</script>"],
            "sources": [
                {
                    "source_id": "web:1",
                    "title": "Official <source>",
                    "url": "https://example.com/research?q=1&safe=yes",
                }
            ],
            "elapsed_seconds": 1.25,
        },
    )

    assert len(events) == 1
    assert events[0]["status"] == "completed"

    rendered = format_research_trace_html(events)
    assert "Discovering &lt;evidence&gt;" in rendered
    assert "&lt;script&gt;alert" in rendered
    assert "<script>" not in rendered
    assert secret not in rendered
    assert "***REDACTED***" in rendered
    assert "1.2s" in rendered
    assert "Worked for 1s" in rendered
    assert "Searched 1 source" in rendered
    assert "Official &lt;source&gt;" in rendered
    assert "https://example.com/research?q=1&amp;safe=yes" in rendered
    assert 'class="activity-drawer"' in rendered
    assert 'data-testid="research-trace" open' not in rendered


def test_running_trace_stays_collapsed_until_the_user_opens_it():
    rendered = format_research_trace_html([], running=True)

    assert 'data-testid="research-trace" open' not in rendered
    assert '<details class="research-trace" data-testid="research-trace">' in rendered
    assert "Working for 0s" in rendered
    assert "Activity summary based on agent actions" in rendered
