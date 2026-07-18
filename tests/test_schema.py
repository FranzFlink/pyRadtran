"""Tests for libRadtran option-schema extraction (pyradtran/schema.py)."""

import json
from pathlib import Path

import pytest

from pyradtran.schema import extract_schema, find_libradtran_root, load_schema

LIBRADTRAN_ROOT = Path("/opt/libRadtran-2.0.6")
HAS_SRC_PY = (LIBRADTRAN_ROOT / "src_py").is_dir()


class TestRootDiscovery:
    def test_root_from_bin_path(self, minimal_config, tmp_path):
        root = tmp_path / "libRadtran-x"
        (root / "src_py").mkdir(parents=True)
        (root / "bin").mkdir()
        minimal_config.paths.libradtran_bin = root / "bin" / "uvspec"
        assert find_libradtran_root(minimal_config) == root

    def test_no_src_py_returns_none(self, minimal_config, tmp_path):
        minimal_config.paths.libradtran_bin = tmp_path / "bin" / "uvspec"
        minimal_config.paths.libradtran_data = tmp_path / "data"
        assert find_libradtran_root(minimal_config) is None

    def test_load_schema_without_root_returns_none(self, minimal_config, tmp_path):
        minimal_config.paths.libradtran_bin = tmp_path / "bin" / "uvspec"
        minimal_config.paths.libradtran_data = tmp_path / "data"
        assert load_schema(minimal_config) is None


@pytest.mark.skipif(not HAS_SRC_PY, reason="local libRadtran src_py not available")
class TestRealExtraction:
    def test_extracts_many_options(self):
        options = extract_schema(LIBRADTRAN_ROOT)
        assert len(options) > 200

    def test_wc_modify_choices_and_dtype(self):
        options = extract_schema(LIBRADTRAN_ROOT)
        wm = options["wc_modify"]
        assert wm["non_unique"] is True
        assert wm["parents"] == ["wc_file"]
        kinds = [t["kind"] for t in wm["tokens"]]
        assert kinds == ["choice", "choice", "value"]
        assert wm["tokens"][0]["choices"] == ["gg", "ssa", "tau", "tau550"]
        assert wm["tokens"][2]["datatype"] == "float"

    def test_albedo_valid_range(self):
        options = extract_schema(LIBRADTRAN_ROOT)
        alb = options["albedo"]
        assert alb["tokens"][0]["valid_range"] == [0.0, 1.0]

    def test_rte_solver_choices_present(self):
        options = extract_schema(LIBRADTRAN_ROOT)
        rs = options["rte_solver"]
        choice_tok = rs["tokens"][0]
        assert choice_tok["kind"] == "choice"
        assert "disort" in choice_tok["choices"]
        assert "twostr" in choice_tok["choices"]

    def test_wc_layer_absent(self):
        # Regression guard: our old parametric cloud path used this
        # nonexistent keyword.
        options = extract_schema(LIBRADTRAN_ROOT)
        assert "wc_layer" not in options

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        import pyradtran.schema as schema_mod

        monkeypatch.setattr(schema_mod, "CACHE_DIR", tmp_path)
        first = load_schema(root=LIBRADTRAN_ROOT)
        cache_files = list(tmp_path.glob("option_schema_*.json"))
        assert len(cache_files) == 1
        # Second call must read the cache, not re-extract
        cache_files[0].write_text(json.dumps({"marker": {"name": "marker"}}))
        second = load_schema(root=LIBRADTRAN_ROOT)
        assert second == {"marker": {"name": "marker"}}
        assert first is not second
