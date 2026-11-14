"""Regression tests for the params / config / era5 / utils bug fixes."""

import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import yaml

import pyradtran
from pyradtran import (
    REGISTRY,
    ConfigurationError,
    InputGenerationError,
    ParamResolver,
    RadiosondeFinder,
    SimulationDefaults,
    ValidationError,
    load_config,
    save_master_config,
)
from pyradtran.config import CloudParameters, PathsConfig
from pyradtran.era5 import cloud_profiles, normalize_era5, select_profile
from pyradtran.params import describe, search_options

RTE_SCHEMA = {
    "rte_solver": {
        "name": "rte_solver",
        "tokens": [{"kind": "choice", "choices": ["disort", "tzs", "brand_new"]}],
    },
    "mol_modify": {
        "name": "mol_modify",
        "non_unique": True,
        "tokens": [
            {"kind": "choice", "choices": ["O3", "H2O"]},
            {"kind": "value", "datatype": "float"},
            {"kind": "choice", "choices": ["DU", "CM_2", "MM"]},
        ],
    },
}


def _config():
    return SimpleNamespace(simulation_defaults=SimulationDefaults())


# ---------------------------------------------------------------- params


class TestParams:
    @pytest.mark.parametrize("solver", ["tzs", "sos", "schwarzschild", "twomaxrnd"])
    def test_rte_solver_accepts_uvspec_solvers_without_schema(self, solver):
        r = ParamResolver(_config(), {"rte_solver": solver}, schema=None)
        assert r.static_params()["rte_solver"][0] == solver

    def test_rte_solver_schema_is_authoritative(self):
        # a solver only the (patched) local schema knows is accepted ...
        ParamResolver(_config(), {"rte_solver": "brand_new"}, schema=RTE_SCHEMA)
        # ... and a typo is rejected, with or without a schema
        for schema in (RTE_SCHEMA, None):
            with pytest.raises(ValidationError, match="rte_solver"):
                ParamResolver(_config(), {"rte_solver": "disrot"}, schema=schema)

    def test_mol_modify_unit_not_duplicated(self):
        spec = REGISTRY["mol_modify O3"]
        assert spec.format_line("320 DU") == "mol_modify O3 320 DU"
        assert spec.format_line(320) == "mol_modify O3 320 DU"
        assert spec.format_line("320") == "mol_modify O3 320 DU"
        r = ParamResolver(_config(), {"mol_modify O3": "320 DU"}, schema=RTE_SCHEMA)
        assert r.static_params()["mol_modify O3"][0] == "320 DU"

    @pytest.mark.parametrize("key", ["albedo", "sza", "zout", "day_of_year"])
    def test_bool_rejected_for_numeric_spec(self, key):
        with pytest.raises(ValidationError, match="expects a number"):
            REGISTRY[key].validate(True)
        with pytest.raises(ValidationError, match="expects a number"):
            ParamResolver(_config(), {key: True}, schema=None)

    def test_false_still_omits_numeric_option(self):
        r = ParamResolver(_config(), {"mol_modify O3": False}, schema=None)
        assert r.static_params()["mol_modify O3"][0] is False

    def test_unknown_dotted_key_raises_with_suggestion(self):
        with pytest.raises(ValidationError, match="simulation_defaults.albedo_value"):
            ParamResolver(
                _config(), {"simulation_defaults.albedo_valu": 1}, schema=None
            )
        with pytest.raises(ValidationError, match="not a configuration field"):
            ParamResolver(_config(), {"nosection.nope": 1}, schema=None)

    def test_resolver_revalidates_simulation_defaults(self):
        with pytest.raises(ValidationError, match="albedo_value"):
            ParamResolver(
                _config(), {"simulation_defaults.albedo_value": 2.0}, schema=None
            )
        config = _config()
        config.simulation_defaults.output_altitudes_km = [5, 0, 5]  # bypasses sorting
        ParamResolver(config, {}, schema=None)
        assert config.simulation_defaults.output_altitudes_km == [0, 5]

    def test_search_and_describe_raise_alike_without_schema(self):
        with patch("pyradtran.params.get_schema", return_value=None):
            with pytest.raises(RuntimeError, match="No local libRadtran") as a:
                describe("albedo", config=_config())
            with pytest.raises(RuntimeError, match="No local libRadtran") as b:
                search_options("albedo", config=_config())
        assert str(a.value) == str(b.value)


# ---------------------------------------------------------------- config


class TestConfig:
    def test_cloud_parameters_accepts_str_paths(self, tmp_path):
        c = CloudParameters(
            enabled=True,
            cloud_source="file",
            cloud_type="mixed",
            wc_file=str(tmp_path / "wc.dat"),
            ic_file=str(tmp_path / "ic.dat"),
        )
        assert c.wc_file == tmp_path / "wc.dat"
        assert c.ic_file == tmp_path / "ic.dat"

    def test_simulation_defaults_validate_rechecks(self):
        sd = SimulationDefaults()
        sd.output_altitudes_km = [3, 1, 3]
        sd.validate()
        assert sd.output_altitudes_km == [1, 3]
        sd.albedo_value = 2.0
        with pytest.raises(ConfigurationError, match="albedo_value"):
            sd.validate()

    def test_configuration_error_is_also_builtin(self):
        assert issubclass(ConfigurationError, pyradtran.PyRadtranError)
        assert issubclass(ConfigurationError, FileNotFoundError)
        assert issubclass(ConfigurationError, ValueError)
        with pytest.raises(ValueError):  # old-style catch keeps working
            SimulationDefaults(source="lunar")

    def test_missing_uvspec_message_is_actionable(self):
        with pytest.raises(ConfigurationError, match="save_master_config") as exc:
            PathsConfig("/nope/uvspec", "/nope")
        assert isinstance(exc.value, FileNotFoundError)

    def test_load_config_without_master_is_actionable(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            with pytest.raises(ConfigurationError, match="save_master_config"):
                load_config()

    def test_load_config_missing_and_broken_yaml(self, tmp_path):
        with pytest.raises(ConfigurationError) as exc:
            load_config(tmp_path / "missing.yaml")
        assert isinstance(exc.value, FileNotFoundError)
        bad = tmp_path / "bad.yaml"
        bad.write_text("paths: [unclosed\n")
        with pytest.raises(ConfigurationError, match="bad.yaml"):
            load_config(bad)

    def test_save_master_config_merges(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            save_master_config(
                "/opt/a/uvspec",
                "/opt/a/data",
                radiosonde_base="/sondes",
                output_dir="/out",
                max_workers=4,
                extra={"simulation_defaults": {"rte_solver": "disort"}},
            )
            path = save_master_config("/opt/b/uvspec", "/opt/b/data")
        content = yaml.safe_load(path.read_text())
        assert content["paths"] == {
            "libradtran_bin": "/opt/b/uvspec",
            "libradtran_data": "/opt/b/data",
            "radiosonde_base": "/sondes",
            "output_dir": "/out",
            "working_dir": "./pyradtran_work",
        }
        assert content["execution"]["max_workers"] == 4
        assert content["simulation_defaults"]["rte_solver"] == "disort"


# ------------------------------------------------------------------ era5


def _grid(lons):
    lats = [80.0, 79.0, 78.0]
    return xr.Dataset(
        {
            "t": (
                ("pressure_level", "latitude", "longitude"),
                np.full((2, len(lats), len(lons)), 260.0),
            )
        },
        coords={
            "pressure_level": [850.0, 1000.0],
            "latitude": lats,
            "longitude": lons,
        },
    )


class TestEra5:
    @pytest.mark.parametrize("unit", ["Pa", "pa", "PA", "Pascal", "pascals"])
    def test_pressure_unit_case_insensitive(self, unit):
        ds = xr.Dataset(
            {"t": ("level", [268.0, 284.0])},
            coords={"level": ("level", [70000.0, 100000.0], {"units": unit})},
        )
        np.testing.assert_allclose(
            normalize_era5(ds).pressure_level.values, [700.0, 1000.0]
        )

    def test_hectopascal_spelling(self):
        ds = xr.Dataset(
            {"t": ("level", [268.0, 284.0])},
            coords={"level": ("level", [700.0, 1000.0], {"units": "hectoPascal"})},
        )
        np.testing.assert_allclose(
            normalize_era5(ds).pressure_level.values, [700.0, 1000.0]
        )

    @pytest.mark.parametrize("missing", ["t", "pressure_level"])
    def test_cloud_profiles_names_missing_variable(self, missing):
        ds = xr.Dataset(
            {
                "t": ("pressure_level", [268.0, 284.0]),
                "clwc": ("pressure_level", [1e-4, 0.0]),
            },
            coords={"pressure_level": [850.0, 1000.0]},
        )
        ds = ds.drop_vars(missing)
        with pytest.raises(InputGenerationError, match=f"'{missing}'"):
            cloud_profiles(ds)

    def test_longitude_wrapped_into_0_360(self):
        ds = _grid([0.0, 90.0, 180.0, 270.0, 350.0])
        assert float(select_profile(ds, 79.0, -10.0).longitude) == 350.0
        assert float(select_profile(ds, 79.0, -90.0).longitude) == 270.0

    def test_longitude_wrapped_into_pm180(self):
        ds = _grid([-170.0, -90.0, 0.0, 90.0, 170.0])
        assert float(select_profile(ds, 79.0, 270.0).longitude) == -90.0
        assert float(select_profile(ds, 79.0, 190.0).longitude) == -170.0

    def test_far_column_warns_but_is_returned(self, caplog):
        ds = _grid([10.0, 11.0, 12.0])
        with caplog.at_level(logging.WARNING, logger="pyradtran.era5"):
            profile = select_profile(ds, 79.0, 11.2)
            assert not caplog.records
            profile = select_profile(ds, 60.0, 11.0)
        assert float(profile.latitude) == 78.0
        assert "more than 1 deg" in caplog.text


# ----------------------------------------------------------------- utils


class TestRadiosondeFinder:
    @pytest.fixture
    def finder(self, tmp_path):
        for name in ["20220328_36000SOD.dat", "20220328_43200SOD.dat"]:
            (tmp_path / name).touch()
        return RadiosondeFinder(tmp_path)

    def test_default_has_no_limit_but_warns(self, finder, caplog):
        with caplog.at_level(logging.WARNING, logger="pyradtran.utils"):
            assert finder.find_closest(datetime(2022, 3, 28, 10, 30)) is not None
            assert not caplog.records
            far = finder.find_closest(datetime(2022, 9, 1))
        assert far.name == "20220328_43200SOD.dat"
        assert "20220328_43200SOD.dat" in caplog.text

    def test_max_hours(self, finder):
        assert finder.find_closest(datetime(2022, 9, 1), max_hours=6) is None
        hit = finder.find_closest(datetime(2022, 3, 28, 13, 0), max_hours=6)
        assert hit.name == "20220328_43200SOD.dat"

    def test_tie_and_edges_unchanged(self, finder):
        assert finder.find_closest(datetime(2022, 3, 28, 11)).name.startswith(
            "20220328_36000"
        )
        assert finder.find_closest(datetime(2022, 3, 27)).name.startswith(
            "20220328_36000"
        )


# ------------------------------------------------------------- top level


def test_top_level_exports():
    for name in ("ValidationError", "RadiosondeError", "RadiosondeAtmosphereGenerator"):
        assert name in pyradtran.__all__
        assert hasattr(pyradtran, name)


def test_quick_start_points_to_online_docs(capsys):
    pyradtran.quick_start()
    out = capsys.readouterr().out
    assert "https://franzflink.github.io/pyRadtran/" in out
    assert "notebooks/" not in out
