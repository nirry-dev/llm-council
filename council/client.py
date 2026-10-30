"""A minimal OpenAI-compatible chat client, built on urllib (no dependencies).

`endpoint` is the API base, e.g. `http://localhost:8080/v1`; this appends
`/chat/completions`. An API key is read from the `api_key` argument or the
`OPENAI_API_KEY` environment variable, and omitted entirely for local servers
that don't want one.
"""
import json
import os
import urllib.error
import urllib.request


class ChatError(RuntimeError):
    """A chat request failed (transport, HTTP status, or malformed response)."""


def chat(endpoint, model, messages, api_key=None, temperature=0.7,
         max_tokens=1024, timeout=120):
    """Send a chat completion request and return the assistant's text."""
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise ChatError(f"{model} @ {endpoint}: HTTP {exc.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ChatError(f"{model} @ {endpoint}: {getattr(exc, 'reason', exc)}") from None

    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise ChatError(f"{model} @ {endpoint}: unexpected response {str(body)[:300]}") from None
