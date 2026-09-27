"""`ash record` must produce something the rest of the pipeline can consume.

That is the whole point of the recorder: the IDM collapses to the class prior
when fitted on the agent's own near-duplicate transitions (val 4.64 against
ln(20)=3.00), and the 24447 recorded human frames are what fixed it (val 0.29).
A recording that `pretrain-idm` cannot read is worth nothing, so the round trip
is what these tests pin - not the recorder's internals.

The other pinned property is that the recorder OBSERVES.  The point of recording
a human is to learn from what they chose; any dispatched key would corrupt the
label it is supposed to be collecting.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from ash.actions.space import ActionSpace
from ash.data.recorder import KEY_TRACKER_JS, HumanSessionConfig, HumanSessionRecorder
from ash.train.demo_mapping import (
    LEGACY_CONTROLS,
    LEGACY_KEY_CODES,
    load_demo,
    map_legacy_masks,
)

CODES = [LEGACY_KEY_CODES[name] for name in LEGACY_CONTROLS]


class _FakeConn:
    """Records every CDP call and refuses any that would move the game."""

    def __init__(self, masks):
        self.will_return = list(masks)
        self.calls: list[str] = []
        self.reads = 0

    def evaluate(self, expr, **_kw):
        self.calls.append(expr)
        if "mask" in expr:
            # Each frame reads the mask twice, before and after the screenshot.
            # A stable key state returns the same value to both reads; modelling
            # them as separate frames would record the wrong masks entirely.
            i = min(self.reads // 2, len(self.will_return) - 1)
            self.reads += 1
            return self.will_return[i] if self.will_return else 0
        return True


class _FakeEnv:
    def __init__(self, frames, masks):
        self.conn = _FakeConn(masks)
        self._frames = list(frames)

    def capture_once(self, **_kw):
        return self._frames.pop(0)


def _recorder(frames, masks, tmp_path, max_frames=0, spill=False, **cfg):
    env = _FakeEnv(frames, masks)
    config = HumanSessionConfig(
        out=tmp_path / "session.npz", size=16, fps=1000.0, max_frames=max_frames,
        session_dir=(tmp_path / "session.npz.session") if spill else None, **cfg)
    rec = HumanSessionRecorder(env, config, LEGACY_CONTROLS, LEGACY_KEY_CODES)
    return env, rec


def _frames(n, size=16, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n)]


def test_recorded_session_is_readable_by_the_training_pipeline(tmp_path):
    """The round trip is the contract: record -> pretrain-idm / --idm-replay."""
    z_up = (1 << LEGACY_CONTROLS.index("z")) | (1 << LEGACY_CONTROLS.index("up"))
    masks = [0, z_up, z_up, 1 << LEGACY_CONTROLS.index("right")]
    env, rec = _recorder(_frames(len(masks)), masks, tmp_path, max_frames=len(masks))
    rec.install()
    rec.run()
    path = rec.save()

    demo = load_demo(path)
    with np.load(path, allow_pickle=True) as raw:
        assert [str(x) for x in raw["control_names"]] == list(LEGACY_CONTROLS)
    assert demo["observations"].shape == (len(masks), 16, 16, 3)
    assert list(demo["control_masks"]) == masks

    # The mapping the whole pipeline runs on must decode the recording.
    space = ActionSpace.minimal()
    index, report = map_legacy_masks(demo["control_masks"], space)
    from ash.actions.space import buttons_from_mask

    buttons = [buttons_from_mask(space.mask_at(int(i))) for i in index]
    # Idle and `right` are members of the curated space; `up+z` is not one of the
    # 20 actions, so it snaps to the nearest - and snapping is counted, not hidden.
    assert report["frames"] == len(masks)
    assert report["exact"] == 2 and report["snapped"] == 2, report
    assert buttons[0] == () and buttons[3] == ("right",)
    assert index.max() < len(space)


def test_recorder_never_dispatches_input(tmp_path):
    """A dispatched key would be recorded as the human's choice.  It never happens."""
    env, rec = _recorder(_frames(4), [0, 1, 0, 1], tmp_path, max_frames=4)
    rec.install()
    rec.run()
    rec.save()
    for call in env.conn.calls:
        assert "Input." not in call and "dispatchKeyEvent" not in call, call


def test_positional_keys_are_never_confused_with_action_names():
    """The mask is over PHYSICAL keys, in one fixed order.

    Z is both jump and confirm and X is both attack and cancel in this game, so a
    `control_names` field that listed game actions would be ambiguous; the
    recorded list is physical keys, and the bit order is the file format.
    """
    assert LEGACY_CONTROLS[:4] == ("up", "down", "left", "right")
    assert LEGACY_CONTROLS[4:9] == ("z", "x", "c", "v", "a")
    assert list(LEGACY_KEY_CODES) == list(LEGACY_CONTROLS)
    assert CODES[:4] == [38, 40, 37, 39]


def test_a_key_pressed_on_a_still_picture_is_down_weighted_not_dropped(tmp_path):
    """A frozen screen with a key down teaches "still picture implies key press".

    The motion distribution is continuous, so any cut-off is arbitrary - the old
    dataset keeps the sample and marks it with a weight, and so does this one.
    """
    still = np.zeros((16, 16, 3), dtype=np.uint8)
    frames = [still.copy() for _ in range(3)]
    masks = [0, 1, 0]                      # key down while nothing moves
    env, rec = _recorder(frames, masks, tmp_path, max_frames=3)
    rec.install()
    rec.run()
    rec.save()
    demo = load_demo(rec.config.out)
    weights = demo["weights"] if "weights" in demo else np.load(rec.config.out)["weights"]
    assert len(weights) == 3               # nothing deleted
    assert weights[1] < 1.0                # the harmful sample is marked
    assert weights[0] == pytest.approx(1.0)


def test_alignment_errors_are_counted_not_hidden(tmp_path):
    """A mask that changes across the screenshot may label the wrong frame."""
    env, rec = _recorder(_frames(2), [1, 2, 3, 4], tmp_path, max_frames=2)
    rec.install()
    rec.run()
    assert rec.alignment_errors == 2


def test_key_tracker_js_encodes_the_documented_bit_order():
    """The JS mask is the format's other half, so it is loaded and exercised."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    script = """
    var handlers = {};
    global.window = {
      addEventListener: function (n, f) { (handlers[n] = handlers[n] || []).push(f); },
    };
    %s
    function fire(name, code) { (handlers[name] || []).forEach(function (f) { f({keyCode: code}); }); }
    var codes = %s;
    fire('keydown', 90);   // z
    fire('keydown', 38);   // up
    console.log(JSON.stringify(window.__ashRec.mask(codes)));
    fire('keyup', 90);
    console.log(JSON.stringify(window.__ashRec.mask(codes)));
    fire('blur', 0);
    console.log(JSON.stringify(window.__ashRec.mask(codes)));
    """ % (KEY_TRACKER_JS, json.dumps(CODES))
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-600:]
    lines = [json.loads(line) for line in out.stdout.strip().splitlines()[-3:]]
    up_bit = 1 << LEGACY_CONTROLS.index("up")
    z_bit = 1 << LEGACY_CONTROLS.index("z")
    assert lines[0] == up_bit | z_bit
    assert lines[1] == up_bit
    assert lines[2] == 0


def test_a_stop_request_ends_the_session_and_still_saves(tmp_path):
    """Nothing is written until save(), so a stop request must not discard data.

    The recorder runs as a background job while the person plays, so it cannot be
    Ctrl-C'd from their terminal.  A default SIGTERM - which kills the process
    outright - would throw the whole recorded session away.
    """
    env, rec = _recorder(_frames(10), [1] * 10, tmp_path, max_frames=10)
    rec.install()
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    frames = rec.run(should_stop=should_stop)
    assert 0 < frames < 10, frames
    path = rec.save()
    assert load_demo(path)["observations"].shape[0] == frames


def test_sigterm_requests_a_stop_instead_of_killing_the_process(tmp_path):
    """A real SIGTERM must leave the process alive with the stop flag set.

    If the handler is ever dropped this test does not fail an assertion - it
    kills the test process, which is exactly the failure the handler prevents.
    """
    import os
    import signal

    from ash.cli.main import _StopFlag

    flag = _StopFlag()
    flag.install()
    try:
        assert not flag.requested()
        os.kill(os.getpid(), signal.SIGTERM)
        assert flag.requested(), "SIGTERM did not reach the stop flag"
    finally:
        flag.restore()
    assert not flag._previous


def test_a_hard_kill_costs_at_most_the_unflushed_tail(tmp_path):
    """The first version wrote one file at the end, so a hard kill lost everything.

    Measured: a backgrounded 16-minute recording was killed and left nothing on
    disk - 9713 frames of the user's own play, gone.  Samples are spilled as they
    arrive now, so `assemble_session` recovers a session whose process is dead.
    """
    from ash.data.recorder import assemble_session

    # flush_every is the loss window: what a kill can cost is exactly the frames
    # still in the file object's buffer, and 20 frames of a cadence-50 recording
    # are all of them - which is why the cadence is a stated knob, not an accident.
    env, rec = _recorder(_frames(20), [1, 2] * 10, tmp_path, max_frames=20,
                         spill=True, flush_every=5)
    rec.install()
    rec.run()
    assert (rec.session_dir / "frames.bin").stat().st_size == 20 * 16 * 16 * 3
    assert (rec.session_dir / "control_masks.bin").stat().st_size == 20 * 8

    # No save() call: the process is "gone".
    out = assemble_session(rec.session_dir, tmp_path / "recovered.npz")
    demo = load_demo(out)
    assert demo["observations"].shape == (20, 16, 16, 3)
    assert list(demo["control_masks"]) == [1, 2] * 10
    with np.load(out, allow_pickle=True) as raw:
        assert [str(x) for x in raw["control_names"]] == list(LEGACY_CONTROLS)
        assert "weights" in raw.files


def test_a_kill_between_the_two_writes_drops_only_the_tail(tmp_path):
    """Frames and masks are two files; a kill can land between them."""
    from ash.data.recorder import assemble_session

    env, rec = _recorder(_frames(4), [1, 2, 3, 4], tmp_path, max_frames=4,
                         spill=True, flush_every=2)
    rec.install()
    rec.run()
    rec.save()                       # closes the handles
    # Reopen the mask file and append one mask with no frame behind it.
    with open(rec.session_dir / "control_masks.bin", "ab") as fh:
        fh.write(np.asarray([7], dtype=np.int64).tobytes())
    out = assemble_session(rec.session_dir, tmp_path / "tail.npz")
    assert len(load_demo(out)["control_masks"]) == 4


def test_spilling_to_a_used_directory_is_refused(tmp_path):
    """Two sessions must never be concatenated into one mask stream."""
    env, rec = _recorder(_frames(2), [1, 2], tmp_path, max_frames=2, spill=True)
    rec.install()
    rec.run()
    second_env, second = _recorder(_frames(2), [3, 4], tmp_path, max_frames=2, spill=True)
    with pytest.raises(RuntimeError, match="already holds a session"):
        second.install()


def test_spilling_does_not_keep_every_frame_in_memory(tmp_path):
    """The point of spilling is also that a long session does not grow RAM."""
    env, rec = _recorder(_frames(30), [1] * 30, tmp_path, max_frames=30, spill=True)
    rec.install()
    rec.run()
    assert rec.frames == [], "spilled frames must not also be retained"
    assert len(rec.masks) == 30, "the masks are small and the summary needs them"
    assert len(rec.motion()) == 30
