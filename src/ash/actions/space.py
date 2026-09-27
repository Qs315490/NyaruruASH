"""Discrete button-mask action space, in the spirit of VPT's action dicts.

A single frame of input is a *set* of held buttons, encoded as a bit mask over a
fixed ordered button list.  The mask is stored as a non-negative int so that it
can be hashed, written to a JSONL file and used directly as a classification
target by the policy head.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

# Ordered button list.  config/keys.yaml maps each of these onto one or more
# concrete key names; keeping the order stable is what makes a mask meaningful
# across processes, checkpoints and log files.
BUTTONS: tuple[str, ...] = (
    "up",
    "down",
    "left",
    "right",
    "jump",     # keyboard: Z / space, pad: A
    "attack",   # keyboard: X, pad: X
    "dash",     # keyboard: C / shift, pad: B/RB
    "special",  # keyboard: V, pad: Y
    "interact", # keyboard: enter / E, pad: Start
    "menu",     # Esc, pad: Back
    # Appended, never inserted: the indices above are baked into recorded
    # masks, checkpoints and log files, so new buttons must extend the tuple.
    "ult",           # keyboard: A - releases the current weapon's ultimate
    "weapon_switch", # keyboard: S - cycles the 咸鱼 weapons
    "cancel",        # keyboard: X - distinct from attack by intent
    "item",          # keyboard: F - uses the selected item (in-map hotkey)
)

BUTTON_INDEX: dict[str, int] = {name: i for i, name in enumerate(BUTTONS)}

# Size of the default action space.  Both model heads (policy and IDM) must be
# built with exactly this many outputs, so it is declared once here instead of
# being repeated as a magic number in each model.  tests/test_action_space.py
# asserts it still equals len(ActionSpace.minimal()), which is what keeps the
# constant from drifting away from the space it describes.
DEFAULT_NUM_ACTIONS = 20


def mask_from_buttons(buttons: Iterable[str]) -> int:
    """Build a bit mask from an iterable of button names."""
    mask = 0
    for name in buttons:
        try:
            bit = BUTTON_INDEX[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError("unknown button %r; known: %s" % (name, ", ".join(BUTTONS))) from exc
        mask |= 1 << bit
    return mask


def buttons_from_mask(mask: int) -> tuple[str, ...]:
    """Inverse of mask_from_buttons."""
    return tuple(name for name, bit in BUTTON_INDEX.items() if mask & (1 << bit))


@dataclass(frozen=True)
class ActionSpace:
    """A finite, ordered set of action masks.

    masks is the actual search/classification space; meta records, for every
    mask, how it was produced ('human', 'heuristic', 'random', ...) which is
    useful when weighting imitation losses.
    """

    masks: tuple[int, ...]
    meta: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.masks:
            raise ValueError("action space must not be empty")
        if len(set(self.masks)) != len(self.masks):
            raise ValueError("action space contains duplicate masks")

    def __len__(self) -> int:
        return len(self.masks)

    def __contains__(self, mask: object) -> bool:
        return mask in self.masks

    @property
    def noop(self) -> int:
        return 0

    def index_of(self, mask: int) -> int:
        """Index of mask inside the space (raises KeyError if absent)."""
        try:
            return self.masks.index(mask)
        except ValueError as exc:
            raise KeyError(
                "mask %d (%s) not in action space" % (mask, buttons_from_mask(mask))
            ) from exc

    def mask_at(self, index: int) -> int:
        return self.masks[index]

    def to_list(self) -> list[int]:
        return list(self.masks)

    @classmethod
    def from_button_names(cls, combos: Sequence[Sequence[str]]) -> ActionSpace:
        return cls(masks=tuple(mask_from_buttons(c) for c in combos))

    @classmethod
    def minimal(cls) -> ActionSpace:
        """A small, hand-picked space that is enough for a first pipeline run.

        Every mask is a legible platformer primitive rather than an arbitrary
        combination, which keeps the classifier head well conditioned.
        """
        combos: list[tuple[str, ...]] = [
            (),
            ("left",),
            ("right",),
            ("jump",),
            ("right", "jump"),
            ("left", "jump"),
            ("attack",),
            ("right", "attack"),
            ("left", "attack"),
            ("dash",),
            ("right", "dash"),
            ("jump", "attack"),
            ("right", "jump", "attack"),
            ("down",),
            ("down", "jump"),
            ("up",),
            # Appended, never inserted: an action index is a recorded label, so
            # the masks above must keep their positions.  These four are the
            # combat and item verbs - the 咸鱼 skill (V, costs SP), the weapon
            # ultimate (A, needs the 必杀 energy bar), cycling the 咸鱼 weapons
            # (S), and using the selected item (F).
            #
            # docs/game-systems.md already flagged S and A as missing: "武器切换
            # 是动作空间的一部分：S 切换 + A 大招目前不在 walker 的动作空间里，
            # 这可能是战斗能力受限的原因之一".  F was deliberately excluded
            # alongside the map key and the fast-load key; the fast-load one is
            # still excluded because it is cheating in a speedrun, while F is the
            # in-map way to eat - i.e. the only healing the policy can do - and
            # the user has asked for it explicitly.
            ("special",),
            ("ult",),
            ("weapon_switch",),
            ("item",),
        ]
        space = cls.from_button_names(combos)
        return cls(masks=space.masks, meta=dict.fromkeys(space.masks, "heuristic"))

    @classmethod
    def from_observed(cls, masks: Iterable[int], *, include_noop: bool = True) -> ActionSpace:
        """Build a space from masks actually seen in a recorded dataset."""
        unique = sorted(set(masks))
        if include_noop and 0 not in unique:
            unique.insert(0, 0)
        return cls(masks=tuple(unique), meta=dict.fromkeys(unique, "observed"))
