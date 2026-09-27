"""The in-page `__ash` agent, loaded from the sibling agent.js.

Keeping the JavaScript in a real .js file makes formatting, linting and static
analysis (eslint, tsc checks, editor tooling) work directly on it, instead of
trapping it inside a Python string literal.
"""

from importlib import resources


def js_source() -> str:
    """Return the injected agent JavaScript source."""
    return (
        resources.files("ash.memory").joinpath("agent.js").read_text(encoding="utf-8")
    )
