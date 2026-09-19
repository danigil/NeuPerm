"""Single source of truth for where the MaleficNet payload `.xor` files live.

Background
----------
This artifact does not distribute the nine live malware samples; it ships inert,
deterministic substitutes under `payloads/synthetic/` (see
`payloads/MANIFEST.sha256`). Every SNR experiment that reads a
`<name>.xor` payload resolves its directory through this module so that:

  * a fresh clone runs out of the box against the safe substitutes, and
  * a researcher who has obtained the real originals (by the hashes in
    MANIFEST.sha256) points at them with one environment variable.

Resolution order for the payload directory:
  1. $NEUPERM_PAYLOAD_DIR   — explicit override (real originals or substitutes)
  2. $MALWARE_DIR           — legacy override kept for existing run recipes
  3. <repo>/payloads/synthetic  — shipped inert substitutes (default)

A requested payload that is absent is a hard error, never a silent skip: the
paper's tables have one row per (model, payload), and skipping a missing payload
would silently drop a row and produce a table of empty cells that looks like a
successful run.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DIR = _REPO_ROOT / "payloads" / "synthetic"


def payload_dir() -> Path:
    """Directory holding the `<name>.xor` payloads (see resolution order above)."""
    for env in ("NEUPERM_PAYLOAD_DIR", "MALWARE_DIR"):
        val = os.environ.get(env)
        if val:
            return Path(val)
    return _DEFAULT_DIR


def require_payload(name: str, directory: Path | None = None) -> Path:
    """Return the path to `<name>.xor`, or raise loudly if it is absent.

    `name` may be given with or without the `.xor` suffix.
    """
    directory = directory or payload_dir()
    fname = name if name.endswith(".xor") else f"{name}.xor"
    path = directory / fname
    if not path.exists():
        raise FileNotFoundError(
            f"payload '{name}' not found at {path}. The artifact ships inert "
            f"substitutes in <repo>/payloads/synthetic/; to reproduce the "
            f"published Table 5 / Figure 3 numbers exactly, obtain the original "
            f"sample by its hash in payloads/MANIFEST.sha256 and set "
            f"NEUPERM_PAYLOAD_DIR (or MALWARE_DIR) to its directory. Refusing to "
            f"skip a missing payload — a dropped row would corrupt the table."
        )
    return path
