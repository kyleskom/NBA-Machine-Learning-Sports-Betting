"""Bounded public-data requests. Credentials and query URLs are not logged."""
import time
import requests


class FeedError(RuntimeError):
    pass


def get_json(url, params=None, headers=None, attempts=3, timeout=20):
    return get_json_with_metadata(url, params, headers, attempts, timeout)[0]


def get_json_with_metadata(url, params=None, headers=None, attempts=3, timeout=20):
    """Fetch JSON and preserve the exact response bytes for source archiving."""
    last = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json(), response.content, dict(response.headers)
        except (requests.RequestException, ValueError) as exc:
            last = exc
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            if code is not None and 400 <= code < 500 and code != 429:
                break
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise FeedError(f'Player feed failed ({type(last).__name__}); existing data retained') from last


def get_bytes(url, params=None, headers=None, attempts=3, timeout=20):
    """Fetch a public binary source and retain its response headers for provenance."""
    last = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.content, dict(response.headers)
        except requests.RequestException as exc:
            last = exc
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            if code is not None and 400 <= code < 500 and code != 429:
                break
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise FeedError(f'Player feed failed ({type(last).__name__}); existing data retained') from last
