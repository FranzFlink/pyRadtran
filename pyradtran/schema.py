"""
libRadtran option-schema extraction.

libRadtran ships a machine-readable description of every ``uvspec``
option in ``<root>/src_py/*_options.py`` (used by its GUI). This module
extracts that description from the *locally installed* libRadtran — so
validation always matches the exact binary being run — and caches it as
JSON under ``~/.pyradtran/``.

The extraction runs in a subprocess to keep libRadtran's GUI modules out
of the pyradtran process. Nothing from libRadtran is redistributed with
pyradtran; the schema is generated on the user's machine from their own
installation.

Schema format (one entry per uvspec option)::

    {
      "name": "wc_modify",
      "group": "Water and ice clouds",
      "help": "Modify water cloud optical properties.",
      "doc": "...",                      # long documentation text
      "non_unique": true,                # option may appear on several lines
      "mandatory": false,
      "parents": ["wc_file"],            # requires one of these
      "childs": [...],
      "tokens": [                        # user-supplied tokens, in order
        {"kind": "choice", "choices": ["gg", "ssa", "tau", "tau550"],
         "optional": false, "file_allowed": false},
        {"kind": "value", "datatype": "float", "valid_range": [0, 1],
         "optional": false}
      ]
    }

See Also
--------
pyradtran.params : Uses the schema for parameter validation.
"""

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Where cached schema files live.
CACHE_DIR = Path.home() / ".pyradtran"

# Runs inside the subprocess with cwd = <root>/src_py.
_EXTRACT_SCRIPT = r'''
import glob, importlib, json, sys

sys.path.insert(0, ".")

TYPE_NAMES = {
    "float": "float", "Double": "float", "int": "int", "Integers": "ints",
    "str": "str", "DataFile": "file", "DataFiles": "files",
    "SignedFloats": "floats", "VariableNumberOfLines": "lines",
    "McPhotons": "int",
}


def dtype_name(dt):
    return TYPE_NAMES.get(getattr(dt, "__name__", str(dt)), "str")


def convert_token(tok):
    kind = type(tok).__name__
    d = getattr(tok, "dict", {}) or {}
    if kind == "addSetting":
        return None  # internal, not a user token
    if kind == "addLogical":
        logicals = d.get("logicals") or []
        choices = [l for l in logicals if isinstance(l, str)]
        file_allowed = len(choices) != len(logicals)
        return {
            "kind": "choice",
            "choices": choices,
            "file_allowed": file_allowed,
            "optional": bool(d.get("optional", False)),
        }
    # addToken and subclasses
    vr = d.get("valid_range")
    if vr is not None:
        try:
            vr = [float(vr[0]), float(vr[1])]
        except (TypeError, ValueError, IndexError):
            vr = None
    return {
        "kind": "value",
        "datatype": dtype_name(d.get("datatype", str)),
        "valid_range": vr,
        "optional": bool(d.get("optional", False)),
    }


options = {}
for path in sorted(glob.glob("*_options.py")):
    modname = path[:-3]
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        print("skip module %s: %s" % (modname, e), file=sys.stderr)
        continue
    for attr in dir(mod):
        if not attr.startswith("setup_"):
            continue
        try:
            grp = getattr(mod, attr)()
        except Exception as e:
            print("skip group %s.%s: %s" % (modname, attr, e), file=sys.stderr)
            continue
        gname = getattr(grp, "group_name", modname)
        for opt in getattr(grp, "options", []):
            try:
                d = getattr(opt, "dict", {}) or {}
                raw_tokens = getattr(opt, "tokens", [])
                tokens = [convert_token(t) for t in raw_tokens]
                # No tokens declared but GUI inputs present: the option
                # takes arguments the schema does not model (e.g.
                # radiosonde) -> signature unknown, skip arity checks.
                unmodeled = (not raw_tokens) and bool(getattr(opt, "gui_inputs", ()))
                options[opt.name] = {
                    "name": opt.name,
                    "group": gname,
                    "help": getattr(opt, "help", "") or "",
                    "doc": (d.get("documentation") or "").strip(),
                    "non_unique": bool(getattr(opt, "non_unique", False)),
                    "mandatory": bool(getattr(opt, "_mandatory", False)),
                    "unmodeled": unmodeled,
                    "parents": [p for p in (d.get("parents") or []) if p != "uvspec"],
                    "childs": list(d.get("childs") or []),
                    "tokens": [t for t in tokens if t is not None],
                }
            except Exception as e:
                print("skip option %s: %s" % (getattr(opt, "name", "?"), e),
                      file=sys.stderr)

json.dump(options, sys.stdout)
'''


def find_libradtran_root(config) -> Optional[Path]:
    """Locate the libRadtran installation root from config paths.

    Tries the parents of ``paths.libradtran_bin`` (``<root>/bin/uvspec``)
    and ``paths.libradtran_data`` (``<root>/data``); returns the first
    candidate that contains ``src_py``.
    """
    candidates = []
    bin_path = getattr(config.paths, "libradtran_bin", None)
    if bin_path:
        p = Path(bin_path)
        candidates += [p.parent.parent, p.parent]
    data_path = getattr(config.paths, "libradtran_data", None)
    if data_path:
        p = Path(data_path)
        candidates += [p.parent, p.parent.parent]
    for cand in candidates:
        if (cand / "src_py").is_dir():
            return cand
    return None


#: Bump when the extracted schema format changes (invalidates caches).
_SCHEMA_FORMAT = "2"


def _cache_key(src_py: Path) -> str:
    """Fingerprint of the src_py option files (name, size, mtime)."""
    h = hashlib.sha1()
    h.update(_SCHEMA_FORMAT.encode())
    h.update(str(src_py).encode())
    for f in sorted(src_py.glob("*_options.py")):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def extract_schema(root: Path) -> Dict[str, Any]:
    """Extract the option schema from ``<root>/src_py`` (no cache).

    Raises
    ------
    RuntimeError
        If the extraction subprocess fails or returns no options.
    """
    src_py = Path(root) / "src_py"
    result = subprocess.run(
        [sys.executable, "-c", _EXTRACT_SCRIPT],
        cwd=src_py,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Schema extraction from {src_py} failed:\n{result.stderr}"
        )
    options = json.loads(result.stdout)
    if not options:
        raise RuntimeError(f"Schema extraction from {src_py} found no options")
    if result.stderr.strip():
        logger.debug(f"Schema extraction notes: {result.stderr.strip()}")
    return options


def load_schema(config=None, root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the option schema for the local libRadtran install, cached.

    Parameters
    ----------
    config : SimulationConfig, optional
        Used to locate the installation when *root* is not given.
    root : pathlib.Path, optional
        libRadtran installation root (the directory containing
        ``src_py``).

    Returns
    -------
    dict or None
        ``{option_name: schema_entry}``, or *None* when no local
        libRadtran source tree can be found (validation then falls back
        to the curated registry only).
    """
    if root is None:
        if config is None:
            return None
        root = find_libradtran_root(config)
    if root is None:
        return None
    src_py = Path(root) / "src_py"
    if not src_py.is_dir():
        return None

    cache_file = CACHE_DIR / f"option_schema_{_cache_key(src_py)}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning(f"Corrupt schema cache {cache_file}; regenerating")

    try:
        options = extract_schema(root)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"Could not extract libRadtran option schema: {e}")
        return None

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(options))
        logger.info(
            f"Cached libRadtran option schema ({len(options)} options) "
            f"at {cache_file}"
        )
    except OSError as e:
        logger.warning(f"Could not write schema cache: {e}")
    return options


__all__ = ["load_schema", "extract_schema", "find_libradtran_root"]
