"""The confirm allow-list must not contain a menu.

`press_ok()` exists for scenes where ok does exactly one thing: continue.  It
was briefly pointed at `Scene_Transport`, on the theory that it was the game's
transition screen needing one ok.  It is actually the TELEPORT DESTINATION
SELECTION MENU: ok there picks a teleport target.  So the agent selected a
destination it never chose and dropped the player onto a damage trap.

That is the same class of failure as pressing ok on the title screen and loading
a save, so the policy is pinned here: the allow-list is empty, and the loaded
agent.js is inspected rather than a copy of the list being trusted.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

AGENT_JS = Path(__file__).resolve().parent.parent / "src" / "ash" / "memory" / "agent.js"

#: Scenes whose ok selects among entries.  Never allow-listable.
MENU_SCENES = (
    "Scene_Title",
    "Scene_Menu",
    "Scene_Item",
    "Scene_Skill",
    "Scene_Equip",
    "Scene_Status",
    "Scene_Options",
    "Scene_File",
    "Scene_Save",
    "Scene_Load",
    "Scene_Shop",
    "Scene_Transport",
)


def _confirm_scenes() -> list[str]:
    """Read V.CONFIRM_SCENES out of the shipped agent.js, via node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    script = (
        "global.window = global;"
        "global.document = {addEventListener: function () {}};"
        "global.navigator = {};"
        "eval(require('fs').readFileSync(%r, 'utf8'));"
        "console.log(JSON.stringify(window.__ash.CONFIRM_SCENES));" % str(AGENT_JS)
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, "agent.js failed to load in node:\n%s" % out.stderr[-600:]
    return json.loads(out.stdout.strip().splitlines()[-1])


#: The scenes whose ok is PROVEN to do exactly one thing.  `Scene_ItemObtain`
#: binds pressOk to popScene (nya_game.js), so ok closes the popup and commits
#: nothing; a live run aborted the whole rollout on it because it was unlisted
#: ("scene 'Scene_ItemObtain' is not Scene_Map gameplay"), stalling the agent
#: every time it picked an item up.
PROVEN = {"Scene_ItemObtain"}


def test_only_proven_popups_are_allow_listed():
    """An entry needs a proof, and the list is exactly the proven set."""
    assert set(_confirm_scenes()) == PROVEN, _confirm_scenes()


def test_an_unproven_popup_is_not_allow_listed():
    """Adding a scene because it looks harmless is the Scene_Transport mistake."""
    for guess in ("Scene_Popup", "Scene_HardGuide", "Scene_AdWaiting",
                  "Scene_NotificationBar", "Scene_Story", "Scene_Staff"):
        assert guess not in _confirm_scenes(), guess


@pytest.mark.parametrize("scene", MENU_SCENES)
def test_teleport_selection_menu_is_not_allow_listed(scene):
    assert scene not in _confirm_scenes(), (
        "%s selects among entries on ok; allow-listing it makes press_ok() a "
        "menu-selection primitive" % scene
    )


def test_press_ok_is_refused_when_the_list_is_empty():
    """With an empty list the exception cannot fire at all."""
    from ash.env.cdp_backend import CdpSpeedrunEnv

    env = CdpSpeedrunEnv(auto_connect=False)
    env.conn = None  # type: ignore[assignment]
    assert env.is_confirm_scene() is False
