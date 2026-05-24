# logger.py
#
# Shared logger for LocalWhisper.
# Writes to both the terminal (so you see it live) and to ivy.log (so you
# can review the full trace after a hang or crash).
#
# Usage in any module:
#   from logger import log
#   log.info("message")
#   log.debug("detail: %s", some_value)
#   log.error("something broke: %s", err)

import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).parent / "ivy.log"

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("ivy")
    if logger.handlers:          # don't add handlers twice on re-import
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── terminal handler (INFO and above) ─────────────────────────────────────
    # Shows concise progress while the agent runs.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # ── file handler (DEBUG and above) ────────────────────────────────────────
    # Full trace including args previews and timing.
    # Append mode: survives multiple runs, so you can diff sessions.
    file_h = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)

    return logger


log = _build_logger()
