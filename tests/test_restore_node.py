"""Reference ids must survive JSON serialization order.

Encoder, decoder and applyInPlace all assign/consume reference ids in
traversal order, so all three must traverse keys in the SAME order.

They did not.  The encoder used Object.keys(value).sort() - a STRING sort -
but JSON.stringify/parse and JS property enumeration both place integer-like
keys FIRST in NUMERIC order.  For keys ["10","2"] the encoder numbered ids
10-then-2 while the serialized text presented 2-then-10, so the decoder
consumed them in the opposite order: {__r:N} resolved to the wrong object,
or to nothing because that id was not created yet.

Reproduced offline before the fix: three aliases of one object came back as
SHARED, undefined, undefined.  That is how a restored Game_Event lost its
methods and killed the game with "eventIsStarting is not a function" on the
44-event map 14.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ash.memory.js_source import js_source

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to execute the injected JS"
)

HARNESS_PATH = Path(__file__).with_name("_refharness.js")


def _run(tmp_path: Path, source: str) -> dict:
    js_file = tmp_path / "inject.js"
    js_file.write_text(source, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(HARNESS_PATH), str(js_file)],
        capture_output=True, text=True, timeout=120, check=False,
        cwd=str(HARNESS_PATH.parent),
    )
    assert proc.returncode == 0, "harness failed: %s\n%s" % (proc.stdout, proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_integer_like_keys_do_not_misalign_reference_ids(tmp_path: Path):
    report = _run(tmp_path, js_source())
    assert report["allSame"], (
        "aliases diverged across the round trip: tags=%s" % (report["tags"],)
    )
    assert report["tags"] == ["SHARED", "SHARED", "SHARED"], (
        "a reference resolved to the wrong object or to nothing: %s"
        % (report["tags"],)
    )


def test_the_test_detects_the_regression(tmp_path: Path):
    """Mutation check: restore the buggy string-sort and confirm failure.

    A regression test that cannot fail is worthless, so the original defect is
    reintroduced here and must be caught.
    """
    source = js_source()
    mutated = source.replace("var keys = canonicalKeys(value);",
                             "var keys = Object.keys(value).sort();")
    assert mutated != source, "could not inject the mutation"
    report = _run(tmp_path, mutated)
    assert not report["allSame"] or report["tags"] != ["SHARED"] * 3, (
        "the mutation was NOT detected - the test cannot catch this bug"
    )
