"""The default test suite must fail fast on accidental external requests."""

import pytest
import requests


def test_unmocked_requests_are_blocked():
    with pytest.raises(pytest.fail.Exception, match="unmocked requests network call"):
        requests.get("https://example.com/should-not-be-called")
