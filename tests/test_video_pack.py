"""The per-video pack layout: one folder per video holding the video, its metadata and artifacts.

Pinned here because the failure mode is silent and expensive: a script that reads the wrong
location gets either nothing (and quietly falls back to a default) or another video's geometry.
Both have happened in this project - a missing `panel_size` fell back to the 720p default and
cropped the wrong region, and a mapping filtered by the built-in cell list silently dropped every
hand-added key.  A pack is meant to make "which file belongs to which video" answerable by looking
at the folder.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.data import video_pack as vp  # noqa: E402

STEM = "TESTVID001"


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """A throwaway data root, so the test never depends on data/ (which is not in the repo)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "video-src").mkdir(parents=True)
    (tmp_path / "data" / "corpus").mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    return tmp_path


def _legacy(sandbox: Path):
    (sandbox / "data" / "video-src" / f"{STEM}.mp4").write_bytes(b"fake video")
    (sandbox / "runs" / f"keycast-cells-{STEM}.json").write_text(json.dumps({
        "video": f"{STEM}.mp4", "panel": [10, 20], "panel_size": [300, 138],
        "style": "bluekey", "cells": {"L": [40, 70]}, "dir_proto": {}}))
    (sandbox / "runs" / f"keycast-mapping-{STEM}.json").write_text(json.dumps({
        "video": f"{STEM}.mp4", "cells": {"L": "left"}}))
    (sandbox / "runs" / f"keycast-labels-{STEM}.npz").write_bytes(b"fake npz")


def test_meta_roundtrip(sandbox):
    meta = {"id": STEM, "panel": [1, 2], "panel_size": [3, 4],
            "mapping": {"L": "left"}, "style": "bright", "cells": {"L": [5, 6]}}
    vp.save_meta(STEM, meta)
    assert vp.meta_path(STEM).exists()
    assert vp.load_meta(STEM) == meta


def test_load_meta_merges_the_two_legacy_files(sandbox):
    """The panel lived in the cells file and the action mapping in the mapping file.

    Both are needed to decode labels, so a reader that consults only one of them is broken - this
    pins that the fallback merges them.
    """
    _legacy(sandbox)
    meta = vp.load_meta(STEM)
    assert meta is not None
    assert meta["panel"] == [10, 20] and meta["panel_size"] == [300, 138]
    assert meta["cells"] == {"L": [40, 70]}
    assert meta["mapping"] == {"L": "left"}
    assert meta["style"] == "bluekey"


def test_load_meta_is_none_when_nothing_exists(sandbox):
    assert vp.load_meta("NOSUCHVIDEO") is None


def test_artifact_prefers_the_pack_and_falls_back(sandbox):
    _legacy(sandbox)
    # before migration: the legacy path is found
    assert vp.artifact(STEM, "labels") == Path("runs") / f"keycast-labels-{STEM}.npz"
    assert vp.artifact(STEM, "frames-30fps") is None          # never produced
    vp.migrate(STEM)
    # after migration: the pack wins, and the file really is there
    got = vp.artifact(STEM, "labels")
    assert got == vp.pack_dir(STEM) / "labels.npz"
    assert got.exists()
    assert vp.artifact(STEM, "video") == vp.pack_dir(STEM) / "video.mp4"


def test_artifact_rejects_an_unknown_kind(sandbox):
    """A typo must raise, not silently return None and look like "not produced yet"."""
    with pytest.raises(KeyError):
        vp.artifact(STEM, "frames-60fps")


def test_migrate_links_instead_of_copying_and_is_idempotent(sandbox):
    """Hard links: a 2 GB video must not become two 2 GB videos just by being packed."""
    _legacy(sandbox)
    first = vp.migrate(STEM)
    assert first["video.mp4"] == "linked"
    src = sandbox / "data" / "video-src" / f"{STEM}.mp4"
    dst = vp.pack_dir(STEM) / "video.mp4"
    assert os.stat(src).st_ino == os.stat(dst).st_ino        # same bytes, not a copy
    again = vp.migrate(STEM)
    assert again["video.mp4"] == "exists"                    # idempotent
    assert len([p for p in vp.list_packs()]) == 1


def test_migrate_writes_the_merged_meta(sandbox):
    _legacy(sandbox)
    vp.migrate(STEM)
    meta = json.loads(vp.meta_path(STEM).read_text())
    assert meta["panel_size"] == [300, 138]
    assert meta["mapping"] == {"L": "left"}
    for key in vp.REQUIRED_META:
        assert key in meta, f"a usable pack must carry {key!r}"


def test_list_packs_only_returns_directories(sandbox):
    _legacy(sandbox)
    vp.migrate(STEM)
    (sandbox / "data" / "videos" / "stray.txt").write_text("not a pack")
    assert vp.list_packs() == [STEM]


def test_game_area_must_be_measured_not_guessed(sandbox):
    """A pack without a game area is usable but incomplete, and must SAY so.

    The game's rectangle differs per video (a 720p capture with a split column puts it somewhere
    else than a 1080p one), so it is measured off a frame; guessing it is what produced the
    "73% of the pixels changed between frames" reading of the middle of a game.
    """
    _legacy(sandbox)
    vp.migrate(STEM)
    meta = vp.load_meta(STEM)
    meta["size"] = [1280, 720]
    vp.save_meta(STEM, meta)
    problems = vp.validate_meta(STEM)
    assert any(vp.GAME_AREA_KEY in p for p in problems)
    assert vp.game_area(STEM) is None


def test_validate_meta_rejects_a_video_field_that_is_not_in_the_pack(sandbox):
    _legacy(sandbox)
    vp.migrate(STEM)
    meta = vp.load_meta(STEM)
    meta["video"] = "SOME_LEGACY_NAME.mp4"        # points at a file that is not there
    meta["game_area"] = [[0, 0], [640, 480]]
    vp.save_meta(STEM, meta)
    problems = vp.validate_meta(STEM)
    assert any("not in the pack" in p for p in problems)
    # after migrate() the field names the real file, so it stops complaining
    meta["video"] = "video.mp4"
    vp.save_meta(STEM, meta)
    assert not [p for p in vp.validate_meta(STEM) if "not in the pack" in p]


def test_validate_meta_checks_game_area_geometry(sandbox):
    _legacy(sandbox)
    vp.migrate(STEM)
    meta = vp.load_meta(STEM)
    meta.update({"size": [1280, 720], "video": "video.mp4"})
    for bad, why in (
        ([[0, 0], [10, 10]], "degenerate"),
        ([[640, 480], [0, 0]], "wrong order"),
        ([[0, 0], [2000, 900]], "outside"),
    ):
        meta["game_area"] = bad
        vp.save_meta(STEM, meta)
        problems = vp.validate_meta(STEM)
        assert problems, "expected a problem for %s" % why
        assert any(vp.GAME_AREA_KEY in p for p in problems)


def test_game_area_returns_corners(sandbox):
    _legacy(sandbox)
    vp.migrate(STEM)
    meta = vp.load_meta(STEM)
    meta["game_area"] = [[300, 0], [1280, 520]]
    vp.save_meta(STEM, meta)
    assert vp.game_area(STEM) == ((300, 0), (1280, 520))


def test_stem_of_survives_absolute_paths(sandbox):
    """The id comes from the pack DIRECTORY, and an already-resolved path must still work.

    Every pack names its file "video.mp4", so deriving the id from the filename calls every video
    "video" - which is what the labeler did after the migration, because main() resolves --video
    and the comparison against the relative VIDEO_ROOT then missed.
    """
    _legacy(sandbox)
    vp.migrate(STEM)
    rel = vp.pack_dir(STEM) / "video.mp4"
    assert vp.stem_of(rel) == STEM
    assert vp.stem_of(rel.resolve()) == STEM            # absolute path, same answer
    assert vp.stem_of(str(rel)) == STEM                 # a plain string too
    assert vp.stem_of("SOMEWHERE/OTHERVID.mp4") == "OTHERVID"   # outside a pack: filename stem


def test_artifact_path_always_lands_in_the_pack(sandbox):
    """Writers must not have a legacy fallback, or new artifacts keep landing in the old place."""
    got = vp.artifact_path(STEM, "labels")
    assert got == vp.pack_dir(STEM) / "labels.npz"
    assert vp.pack_dir(STEM).is_dir()                    # created for the writer
    with pytest.raises(KeyError):
        vp.artifact_path(STEM, "nonsense")
