# tests/helpers.py
"""Shared test helpers for pyradtran tests."""

from pathlib import Path


def has_libradtran() -> bool:
    """True when the master config points to an existing uvspec binary & data dir.

    Reads ``~/.pyradtran/config.yaml`` via :func:`pyradtran.config.load_config`.
    """
    try:
        from pyradtran.config import load_config

        cfg = load_config()
        bin_path = cfg.paths.libradtran_bin
        data_path = cfg.paths.libradtran_data
        return Path(bin_path).is_file() and Path(data_path).is_dir()
    except Exception:
        return False
