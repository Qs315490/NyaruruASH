"""Device resolution shared by the loop, the trainers and the CLI.

Every component takes `device: str | None` and calls resolve_device(), because
the CLI passes `--device` straight through and its default is None.  Passing
None to torch.device() raises

    TypeError: device() received an invalid combination of arguments

which is a configuration error that should never reach the user as a crash.
`None` and `"auto"` both mean "pick the best available device".
"""

from __future__ import annotations

import torch


def resolve_device(device: str | None = None) -> torch.device:
    """Return a torch.device for `device`; None/"auto" picks cuda when present.

    ROCm builds report through torch.cuda, so the same branch covers AMD.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
