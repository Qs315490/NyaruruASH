"""Translate the legacy human-demo control masks into this project's buttons.

`vpt-amd-package/data/idm-human.npz` holds 24447 frames of recorded human play
with a 17-bit control mask per frame.  That is the only action-labelled data this
project has, and it is what makes "learn to leave the room" possible at all -
random self-play was measured not to produce the needed dynamics (IDM value
pinned at ln(num_classes) from 220 up to 1060 transitions).

The legacy list and the 17/14 button sets do not match, so the translation is
explicit rather than positional:

- five legacy keys were dropped on purpose by this project and have no button:
  `m` (map) and `j`/`i`/`k`/`l` (camera pan).  A frame whose only presses are
  those becomes noop, so the count of such frames is reported instead of hidden.
- `x` is ONE physical key that this project models twice, as `attack` (the
  in-map verb) and `cancel` (the menu verb).  A demo press is translated to
  `attack`; `cancel` is only ever produced by the menu-escape path.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ash.actions.space import ActionSpace, buttons_from_mask, mask_from_buttons

#: The legacy control list, in bit order (from `control_names` in the npz).
LEGACY_CONTROLS: tuple[str, ...] = (
    "up", "down", "left", "right", "z", "x", "c", "v", "a", "escape", "f",
    "m", "enter", "j", "i", "k", "l",
)

#: Browser keyCode for each legacy control, in the same order.
#:
#: These are physical keys, not game actions: the recorder captures what the
#: player pressed, and mapping keys to actions is `LEGACY_TO_BUTTON`'s job.  The
#: game binds several actions to one key (Z is jump *and* confirm, X is attack
#: *and* cancel), so an IDM cannot be told the intent - only which keys were down.
#: Values match RPG Maker MZ's key mapper and the recordings in `data/idm-human/`.
LEGACY_KEY_CODES: dict[str, int] = {
    "up": 38, "down": 40, "left": 37, "right": 39,
    "z": 90, "x": 88, "c": 67, "v": 86, "a": 65,
    "escape": 27, "f": 70, "m": 77, "enter": 13,
    "j": 74, "i": 73, "k": 75, "l": 76,
}

#: Legacy name -> this project's button.  None means "deliberately unsupported".
LEGACY_TO_BUTTON: dict[str, str | None] = {
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "z": "jump",        # z is both jump and the ok key
    "x": "attack",      # x is both attack and the cancel key
    "c": "dash",
    "v": "special",
    "a": "ult",
    "escape": "menu",
    "f": "item",
    "enter": "interact",
    "m": None,          # map key: excluded
    "j": None,          # camera pan: excluded
    "i": None,
    "k": None,
    "l": None,
}

DROPPED_LEGACY = tuple(n for n, b in LEGACY_TO_BUTTON.items() if b is None)


def legacy_mask_to_buttons(mask: int) -> tuple[str, ...]:
    """Legacy 17-bit mask -> this project's button names, dropping unsupported keys."""
    out = []
    for bit, name in enumerate(LEGACY_CONTROLS):
        if mask & (1 << bit):
            button = LEGACY_TO_BUTTON[name]
            if button is not None:
                out.append(button)
    return tuple(out)


def legacy_mask_to_our_mask(mask: int) -> int:
    """Legacy 17-bit mask -> this project's bit mask over BUTTONS."""
    return mask_from_buttons(legacy_mask_to_buttons(mask))


def _distance(space_buttons: tuple[str, ...], want: frozenset[str]) -> int:
    """Symmetric difference, i.e. buttons wrong in either direction."""
    return len(set(space_buttons) ^ want)


def demo_replay_trajectories(
    path: str | Path,
    space: ActionSpace,
    *,
    steps: int = 4000,
    seed: int = 0,
) -> list[dict]:
    """A bounded, per-round sample of the human demonstrations, as trajectories.

    The bootstrap fits the IDM on the agent's own transitions, and from a
    standing start those are near-duplicates in one small room.  Measured: a
    demonstration-pretrained IDM (val 0.29 against ln(20)=3.00) was driven back
    to val 3.16 - the class prior - by three epochs on 220 such transitions.
    Mixing the demonstrations back in is what keeps the learned dynamics instead
    of overwriting them, which is the VPT recipe (pretrain on human data, then
    keep training on the union).

    Sampling rather than loading all 24434 transitions keeps a round's cost
    bounded; the seed is the round index, so successive rounds replay different
    parts and the whole set is eventually seen.
    """
    return sample_replay(load_demo(path), space, steps=steps, seed=seed)


DEMO_KEYS: tuple[str, ...] = ("observations", "control_masks", "episode_ids")


def pack_demo(src: str | Path, dst: str | Path | None = None) -> Path:
    """Rewrite the demo npz as uncompressed .npy files so a run can mmap them.

    Materializing the npz costs 1.2 GB for `observations` alone, and that was
    enough to have the host OOM-kill a round when the corpus videos, the KDM and
    the models were already resident.  An uncompressed directory can be read with
    `mmap_mode="r"`, so a round only faults in the windows it samples.
    """
    import numpy as np

    src = Path(src)
    dst = Path(dst) if dst is not None else src.with_suffix("")
    dst.mkdir(parents=True, exist_ok=True)
    with np.load(src, allow_pickle=True) as data:
        for key in DEMO_KEYS:
            np.save(dst / f"{key}.npy", data[key])
    return dst


def load_demo(path: str | Path) -> dict:
    """Read the demonstrations, memory-mapped when the packed form exists."""
    import numpy as np

    path = Path(path)
    packed = path.with_suffix("")
    if packed.is_dir() and all((packed / f"{k}.npy").exists() for k in DEMO_KEYS):
        return {k: np.load(packed / f"{k}.npy", mmap_mode="r") for k in DEMO_KEYS}
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in DEMO_KEYS}


def sample_replay(demo: dict, space: ActionSpace, *, steps: int = 4000,
                  seed: int = 0) -> list[dict]:
    """A bounded window of the demonstrations per episode, chosen by `seed`."""
    import numpy as np

    obs = demo["observations"]
    index, _ = map_legacy_masks(demo["control_masks"], space)
    episodes = demo["episode_ids"]
    ids = list(np.unique(episodes))
    if not ids or steps <= 0:
        return []
    per_episode = max(2, int(steps) // len(ids))
    rng = np.random.default_rng(int(seed))
    out: list[dict] = []
    for episode in ids:
        rows = np.flatnonzero(episodes == episode)
        length = len(rows)
        if length < 3:
            continue
        take = min(per_episode, length)
        start = int(rng.integers(0, length - take + 1)) if length > take else 0
        window = rows[start:start + take]
        frames = obs[window]
        # The mask recorded at t+1 is the label for the pair (t, t+1): the legacy
        # trainer's convention, and the one the recording was made against.
        actions = index[window][1:].astype(np.int64)
        out.append({"obs": frames, "act": actions})
    return out


def map_legacy_masks(masks: np.ndarray, space: ActionSpace) -> tuple[np.ndarray, dict]:
    """Map every legacy mask to an index of `space`, nearest match when inexact.

    A demo press combination need not exist in the curated space (`up+jump`,
    `left+right`).  Snapping to the nearest member keeps the label as close as
    the action space allows, and the report says how often that happened so an
    approximation is never mistaken for the human's actual input.
    """
    masks = np.asarray(masks)
    space_sets = [(i, buttons_from_mask(space.mask_at(i)), frozenset(buttons_from_mask(space.mask_at(i))))
                  for i in range(len(space))]
    exact: dict[int, int] = {}
    for i, _, bs in space_sets:
        exact[bs] = i
    out = np.empty(len(masks), dtype=np.int64)
    counts = Counter()
    dropped_only = 0
    for row, mask in enumerate(masks):
        want_list = legacy_mask_to_buttons(int(mask))
        want = frozenset(want_list)
        if int(mask) and not want_list:
            # The human pressed something, and every press was a key this
            # project cannot send.  The mapped label is noop, which is a valid
            # space member, so this must be counted before the exact-match path
            # or the loss would be invisible.
            dropped_only += 1
        if want in exact:
            out[row] = exact[want]
            counts["exact"] += 1
            continue
        best_i, best_d = None, 10**6
        for i, _buttons, bs in space_sets:
            d = _distance(tuple(bs), want)
            if d < best_d or (d == best_d and best_i is not None
                              and len(bs) < len(space_sets[best_i][2])):
                best_i, best_d = i, d
        out[row] = best_i
        counts["snapped"] += 1
        counts["snap_distance_%d" % best_d] += 1
    report = {
        "frames": int(len(masks)),
        "distinct_legacy_masks": int(len(set(masks.tolist()))),
        "exact": counts["exact"],
        "snapped": counts["snapped"],
        "snap_distances": {k: v for k, v in sorted(counts.items())
                           if k.startswith("snap_distance_")},
        "frames_whose_only_presses_were_dropped_keys": dropped_only,
        "dropped_legacy_keys": DROPPED_LEGACY,
    }
    return out, report
