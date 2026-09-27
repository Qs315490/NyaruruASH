"""Loading of generated data assets, with a loud-but-once failure mode.

Some files under `reports/` are INPUTS rather than build artefacts:

    world-cells.json        vpt.world.topology       (hard: raises)
    savepoints.json         vpt.planner.death        (soft: degrades to [])
    transport-portals.json  scripts/route_walker.py  (soft: degrades to [])

None of them are committed - they are regenerated from the running game with
`scripts/extract_runtime_assets.py`.  The soft ones degrade to an empty result
so a run continues, but degrading silently made every map look savepoint-free
and quietly disabled the heal routing.  Each missing asset is therefore reported
once per process.
"""
from __future__ import annotations

from pathlib import Path

from ash.utils.logging import get_logger

_log = get_logger(__name__)

#: Paths already reported, so a per-map call in a loop warns only once.
_warned: set[str] = set()


def warn_missing_asset_once(kind: str, path: str | Path, exc: BaseException) -> None:
    """Report a missing generated asset once per process."""
    key = "%s:%s" % (kind, path)
    if key in _warned:
        return
    _warned.add(key)
    _log.warning(
        "%s asset is missing or unreadable (%s): %s. This degrades quietly - "
        "regenerate it with: uv run python scripts/extract_runtime_assets.py",
        kind,
        path,
        exc,
    )


def reset_warnings() -> None:
    """Forget which assets were reported (used by tests)."""
    _warned.clear()
