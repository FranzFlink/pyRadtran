r"""
libRadtran option-schema extraction.

libRadtran ships a machine-readable description of every ``uvspec``
option in ``<root>/src_py/*_options.py`` (used by its GUI). This module
extracts that description from the locally installed libRadtran, so
validation matches the binary being run, and caches it as JSON under
``~/.pyradtran/``.

.. autosummary::

   load_schema
   extract_schema
   find_libradtran_root

Most users never call this module directly. The schema is what makes
``params={"wc_modfy": ...}`` fail with a "did you mean ``wc_modify``?"
before any simulation starts, and what
:func:`~pyradtran.params.describe` and
:func:`~pyradtran.params.search_options` print.

The extraction runs in a subprocess to keep libRadtran's GUI modules out
of the pyradtran process. Nothing from libRadtran is redistributed; the
schema is generated on the user's machine from their own installation.

Notes
-----
**Cache.** The schema is stored as
``~/.pyradtran/option_schema_<fingerprint>.json`` (:data:`CACHE_DIR`).
The fingerprint is a hash of the ``src_py`` path and the name, size and
modification time of every ``*_options.py`` file, so switching to
another libRadtran install, or patching the current one, regenerates
the schema. Only the file for the current fingerprint is kept. Deleting
it is safe; it is regenerated on the next use.

**Structure.** The schema is a ``dict`` mapping each option name to one
entry. A real entry from libRadtran 2.0.6 (``doc`` shortened)::

    {
      "name": "wc_modify",
      "group": "Water and ice clouds",
      "help": "Modify water cloud optical properties.",
      "doc": "Modify water cloud optical properties.\n \\fcode{ ...",
      "non_unique": true,
      "mandatory": false,
      "unmodeled": false,
      "parents": ["wc_file"],
      "childs": [],
      "tokens": [
        {"kind": "choice", "choices": ["gg", "ssa", "tau", "tau550"],
         "file_allowed": false, "optional": false},
        {"kind": "choice", "choices": ["set", "scale"],
         "file_allowed": false, "optional": false},
        {"kind": "value", "datatype": "float", "valid_range": null,
         "optional": false}
      ]
    }

``name``
    The uvspec option keyword.
``group``
    Section of the libRadtran manual / GUI, e.g. ``"Surface"``.
``help``, ``doc``
    One-line help and the long documentation (LaTeX-flavoured).
``non_unique``
    The option may appear on several input lines (``wc_modify``,
    ``mol_modify``, ``crs_model``, ...). A ``params`` list on such an
    option becomes one line per element.
``mandatory``
    libRadtran's own "must be present" flag.
``unmodeled``
    The option takes arguments that libRadtran's description does not
    spell out (e.g. ``radiosonde``): only the name is validated.
``parents``, ``childs``
    Options this one requires / enables. Informational: shown by
    :func:`~pyradtran.params.describe`, not enforced.
``tokens``
    The values the user supplies after the keyword, in order. A
    ``"choice"`` token lists the allowed ``choices``
    (``file_allowed`` means a file name is accepted as well); a
    ``"value"`` token has a ``datatype`` (``float``, ``int``, ``str``,
    ``file`` or one of the variable-length types ``floats``, ``ints``,
    ``files``, ``lines``) and, for numbers, an inclusive
    ``valid_range`` or ``null``. ``optional`` tokens may be left out.
    An empty list is a flag option such as ``aerosol_default``.

Examples
--------
>>> from pyradtran import load_config
>>> from pyradtran.schema import load_schema
>>> schema = load_schema(load_config())  # doctest: +SKIP
>>> len(schema)  # doctest: +SKIP
244
>>> schema["albedo"]["tokens"]  # doctest: +SKIP
[{'kind': 'value', 'datatype': 'float', 'valid_range': [0.0, 1.0], 'optional': False}]

See Also
--------
pyradtran.params : Uses the schema for parameter validation.
pyradtran.params.describe : Human-readable view of one schema entry.
"""

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Directory holding the cached schema files
#: (``~/.pyradtran/option_schema_<fingerprint>.json``).
CACHE_DIR = Path.home() / ".pyradtran"

# Runs inside the subprocess with cwd = <root>/src_py.
_EXTRACT_SCRIPT = r"""
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
"""


def find_libradtran_root(config) -> Optional[Path]:
    """Locate the libRadtran installation root from config paths.

    The root is the directory that contains ``src_py`` (libRadtran's
    option descriptions), i.e. a source/build tree rather than a bare
    ``bin`` + ``data`` installation.

    Parameters
    ----------
    config : SimulationConfig
        Only ``config.paths.libradtran_bin`` and
        ``config.paths.libradtran_data`` are read; either may be missing
        or ``None``.

    Returns
    -------
    pathlib.Path or None
        The first of these candidates that has a ``src_py``
        sub-directory, in this order: two and one levels above
        ``libradtran_bin`` (``<root>/bin/uvspec`` or ``<root>/uvspec``),
        then one and two levels above ``libradtran_data``
        (``<root>/data`` or ``<root>/share/data``). ``None`` when no
        candidate qualifies.

    See Also
    --------
    load_schema : Uses the root to extract and cache the schema.

    Examples
    --------
    A fake installation tree (any object with a ``paths`` attribute
    works as *config*):

    >>> import tempfile
    >>> from pathlib import Path
    >>> from types import SimpleNamespace
    >>> from pyradtran.schema import find_libradtran_root
    >>> tmp = Path(tempfile.mkdtemp())
    >>> (tmp / "src_py").mkdir()
    >>> paths = SimpleNamespace(libradtran_bin=tmp / "bin" / "uvspec")
    >>> find_libradtran_root(SimpleNamespace(paths=paths)) == tmp
    True

    Without a ``src_py`` directory there is no root, and schema
    validation is switched off:

    >>> paths = SimpleNamespace(libradtran_bin=tmp / "elsewhere" / "bin" / "uvspec")
    >>> print(find_libradtran_root(SimpleNamespace(paths=paths)))
    None

    With a real configuration:

    >>> from pyradtran import load_config
    >>> find_libradtran_root(load_config())  # doctest: +SKIP
    PosixPath('/opt/libRadtran-2.0.6')
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

    Imports every ``*_options.py`` module of libRadtran's ``src_py``
    directory in a separate Python subprocess (same interpreter, working
    directory ``src_py``, 120 s timeout) and converts the option objects
    to plain JSON-compatible dictionaries. Modules, option groups or
    single options that fail to import are skipped; the skip notes are
    logged at ``DEBUG`` level.

    Parameters
    ----------
    root : str or pathlib.Path
        libRadtran installation root (the directory containing
        ``src_py``).

    Returns
    -------
    dict
        ``{option_name: schema_entry}``; see the module docstring of
        :mod:`pyradtran.schema` for the entry structure.

    Raises
    ------
    RuntimeError
        If the extraction subprocess exits with an error or finds no
        options.
    FileNotFoundError
        If ``<root>/src_py`` does not exist.
    subprocess.TimeoutExpired
        If the extraction takes longer than 120 s.

    See Also
    --------
    load_schema : Cached variant that returns ``None`` instead of raising.

    Examples
    --------
    >>> from pyradtran.schema import extract_schema
    >>> schema = extract_schema("/opt/libRadtran-2.0.6")  # doctest: +SKIP
    >>> len(schema)  # doctest: +SKIP
    244
    >>> sorted(schema["wc_modify"])  # doctest: +SKIP
    ['childs', 'doc', 'group', 'help', 'mandatory', 'name', 'non_unique', 'parents', 'tokens', 'unmodeled']
    >>> schema["sza"]["tokens"]  # doctest: +SKIP
    [{'kind': 'value', 'datatype': 'float', 'valid_range': [0.0, 180.0], 'optional': False}]
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
        raise RuntimeError(f"Schema extraction from {src_py} failed:\n{result.stderr}")
    options = json.loads(result.stdout)
    if not options:
        raise RuntimeError(f"Schema extraction from {src_py} found no options")
    if result.stderr.strip():
        logger.debug(f"Schema extraction notes: {result.stderr.strip()}")
    return options


def load_schema(config=None, root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the option schema for the local libRadtran install, cached.

    The first call for an installation runs :func:`extract_schema` and
    writes ``~/.pyradtran/option_schema_<fingerprint>.json``; later calls
    (in any process) read that file. A changed installation is
    re-extracted and the stale cache file removed (see the cache note in
    :mod:`pyradtran.schema`).

    A missing or broken installation never raises. The function returns
    ``None`` (with a logged warning if the extraction fails) and
    parameter validation falls back to the curated
    :data:`~pyradtran.params.REGISTRY`.

    Parameters
    ----------
    config : SimulationConfig, optional
        Used to locate the installation (via
        :func:`find_libradtran_root`) when *root* is not given.
    root : str or pathlib.Path, optional
        libRadtran installation root (the directory containing
        ``src_py``). Takes precedence over *config*.

    Returns
    -------
    dict or None
        ``{option_name: schema_entry}`` (see the module docstring of
        :mod:`pyradtran.schema`), or ``None`` when neither *config* nor
        *root* is given, no ``src_py`` directory can be found, or the
        extraction fails.

    See Also
    --------
    pyradtran.params.get_schema : Same, memoised in-process per root.
    pyradtran.params.describe : Pretty-print one entry.
    extract_schema : The uncached extraction.

    Notes
    -----
    A cache file that cannot be read or parsed is regenerated; a cache
    directory that cannot be written only produces a warning (the schema
    is then re-extracted in every new process).

    Examples
    --------
    Without an installation:

    >>> from pyradtran.schema import load_schema
    >>> print(load_schema())
    None
    >>> print(load_schema(root="/no/such/libRadtran"))
    None

    With a working configuration (libRadtran 2.0.6):

    >>> from pyradtran import load_config
    >>> schema = load_schema(load_config())  # doctest: +SKIP
    >>> len(schema)  # doctest: +SKIP
    244
    >>> schema["wc_modify"]["tokens"][0]  # doctest: +SKIP
    {'kind': 'choice', 'choices': ['gg', 'ssa', 'tau', 'tau550'], 'file_allowed': False, 'optional': False}
    >>> schema["wc_modify"]["non_unique"]  # doctest: +SKIP
    True
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
        # keep only the current fingerprint; stale caches never expire
        # on their own
        for old in CACHE_DIR.glob("option_schema_*.json"):
            if old != cache_file:
                old.unlink(missing_ok=True)
        logger.info(
            f"Cached libRadtran option schema ({len(options)} options) "
            f"at {cache_file}"
        )
    except OSError as e:
        logger.warning(f"Could not write schema cache: {e}")
    return options


__all__ = ["load_schema", "extract_schema", "find_libradtran_root"]
