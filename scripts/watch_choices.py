"""Print a dialogue's option texts as they appear. Read-only.

The safety layer refuses a choice because the policy's jump key is the ok key and
would commit whichever option is highlighted.  Allowing ordinary dialogue while
still refusing the difficulty pick needs the option texts, and this is how they
are obtained from the running game rather than guessed from packed data files.

It installs the agent (so `V.safety()` exists) and resumes the engine ticker, but
dispatches nothing: run it, trigger the dialogue in game, and read the strings.

    uv run python scripts/watch_choices.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ash.env.cdp_backend import CdpSpeedrunEnv  # noqa: E402


def main() -> int:
    env = CdpSpeedrunEnv(drive="realtime", resize=(128, 128))
    env.connect()
    env.install(seed=0)
    print("watching for dialogue choices; trigger the dialogue in game (Ctrl-C stops)")
    seen: tuple = ()
    try:
        while True:
            info = env.safety() or {}
            current = tuple(info.get("choices") or ())
            if current and current != seen:
                seen = current
                print("[%s] scene=%s busy=%s awaiting=%s choices=%r"
                      % (time.strftime("%H:%M:%S"), info.get("scene"),
                         info.get("messageBusy"), info.get("awaitingChoice"), list(current)))
            elif not current:
                seen = ()
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
