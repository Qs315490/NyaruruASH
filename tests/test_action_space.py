"""The action space is a recorded label space: append only.

An action index is written into trajectories, log files and checkpoints, so
extending the space must never move an existing entry.  The four combat/item
verbs were appended after the original sixteen for exactly that reason, and the
button tuple is subject to the same rule because a mask is a bit over its order.
"""

from __future__ import annotations

from ash.actions.space import (
    BUTTONS,
    BUTTON_INDEX,
    DEFAULT_NUM_ACTIONS,
    ActionSpace,
    buttons_from_mask,
    mask_from_buttons,
)

#: The buttons that already existed, in the order they already had.  Their
#: indices are baked into every mask ever recorded.
ORIGINAL_BUTTONS = (
    "up", "down", "left", "right", "jump", "attack", "dash", "special",
    "interact", "menu", "ult", "weapon_switch", "cancel",
)

#: The original sixteen masks, in their original order.
ORIGINAL_MASKS = (
    (), ("left",), ("right",), ("jump",), ("right", "jump"), ("left", "jump"),
    ("attack",), ("right", "attack"), ("left", "attack"), ("dash",),
    ("right", "dash"), ("jump", "attack"), ("right", "jump", "attack"),
    ("down",), ("down", "jump"), ("up",),
)

#: Appended for this game: the 咸鱼 skill (V, costs SP), the weapon ultimate
#: (A, needs the 必杀 energy bar), cycling the 咸鱼 weapons (S), and using the
#: selected item (F) - the only in-map way to heal.
NEW_VERBS = (("special",), ("ult",), ("weapon_switch",), ("item",))


def test_button_indices_are_append_only():
    for i, name in enumerate(ORIGINAL_BUTTONS):
        assert BUTTON_INDEX[name] == i, (
            "%s moved to bit %d; every recorded mask just changed meaning"
            % (name, BUTTON_INDEX[name])
        )
    assert BUTTONS[: len(ORIGINAL_BUTTONS)] == ORIGINAL_BUTTONS
    assert BUTTONS[-1] == "item"


def test_existing_masks_keep_their_positions():
    space = ActionSpace.minimal()
    for i, combo in enumerate(ORIGINAL_MASKS):
        assert buttons_from_mask(space.mask_at(i)) == combo, (
            "action index %d changed meaning" % i
        )


def test_the_four_verbs_are_appended():
    space = ActionSpace.minimal()
    tail = [buttons_from_mask(space.mask_at(i))
            for i in range(len(ORIGINAL_MASKS), len(space))]
    assert tail == [("special",), ("ult",), ("weapon_switch",), ("item",)]
    for combo in NEW_VERBS:
        mask = mask_from_buttons(combo)
        assert mask in space, "%s is not in the action space" % (combo,)
        assert space.index_of(mask) >= len(ORIGINAL_MASKS)


def test_head_width_matches_the_space():
    """The one constant both model heads default to must equal the real space."""
    assert len(ActionSpace.minimal()) == DEFAULT_NUM_ACTIONS == 20


def test_the_new_keys_are_bound_in_the_keymap():
    """A space entry with no key behind it is a silent no-op action."""
    from ash.config import load_game_config

    keymap = load_game_config().keymap
    for name in ("special", "ult", "weapon_switch", "item"):
        binding = keymap.get(name)
        assert binding is not None and binding.names, "%s has no key binding" % name
    assert "f" in [n.lower() for n in load_game_config().keymap["item"].names]
