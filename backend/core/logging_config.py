"""
TerraMind - Structured logging configuration.

Provides a pre-configured logger with console + file handlers.
"""
import logging
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str = "terramind", level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with console and file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger          # avoid duplicate handlers on re-import

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler - use UTF-8 wrapper to avoid cp1252 crashes on Windows
    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    ch = logging.StreamHandler(utf8_stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(_LOG_DIR / "terramind.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

log = get_logger()
