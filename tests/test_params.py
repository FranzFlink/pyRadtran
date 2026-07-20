"""Tests for the parameter registry and resolver (pyradtran/params.py)."""

import pytest

from pyradtran.exceptions import ValidationError
from pyradtran.params import CONFIG_FIELD_MAP, REGISTRY, ParamSpec, Var


class TestVar:
    def test_var_holds_name(self):
        v = Var("sfc_albedo")
        assert v.name == "sfc_albedo"

    def test_var_is_frozen(self):
        v = Var("x")
        with pytest.raises(Exception):
            v.name = "y"

    def test_var_equality(self):
        assert Var("a") == Var("a")
        assert Var("a") != Var("b")


class TestRegistry:
    def test_core_parameters_present(self):
        for key in [
            "albedo",
            "sur_temperature",
            "sza",
            "zout",
            "brdf_rpv_type",
            "wavelength",
            "mol_modify O3",
            "mol_modify H2O",
            "rte_solver",
            "mol_abs_param",
            "umu",
            "output_user",
            "source",
            "day_of_year",
        ]:
            assert key in REGISTRY, f"missing registry entry: {key}"

    def test_albedo_range_validation(self):
        spec = REGISTRY["albedo"]
        spec.validate(0.5)  # no raise
        with pytest.raises(ValidationError):
            spec.validate(1.5)
        with pytest.raises(ValidationError):
            spec.validate(-0.1)

    def test_sur_temperature_positive(self):
        with pytest.raises(ValidationError):
            REGISTRY["sur_temperature"].validate(-5.0)

    def test_source_choices(self):
        REGISTRY["source"].validate("solar")
        with pytest.raises(ValidationError):
            REGISTRY["source"].validate("lunar")

    def test_format_line_default(self):
        assert REGISTRY["albedo"].format_line(0.85) == "albedo 0.85"

    def test_format_line_mol_modify(self):
        assert REGISTRY["mol_modify O3"].format_line(300.0) == "mol_modify O3 300.0 DU"

    def test_format_line_brdf_type_is_int(self):
        assert REGISTRY["brdf_rpv_type"].format_line(7.0) == "brdf_rpv_type 7"

    def test_string_values_bypass_numeric_validation(self):
        # Raw pass-through strings (escape hatch) must not be range-checked
        REGISTRY["zout"].validate("0.0 1.0 120.0")


class TestConfigFieldMap:
    def test_known_mappings(self):
        assert CONFIG_FIELD_MAP["albedo"] == "albedo_value"
        assert CONFIG_FIELD_MAP["sur_temperature"] == "surface_temperature_k"
        assert CONFIG_FIELD_MAP["sza"] == "sza"

    def test_all_map_keys_in_registry(self):
        for key in CONFIG_FIELD_MAP:
            assert key in REGISTRY


from pyradtran.params import (
    PROV_LITERAL,
    PROV_UNVALIDATED,
    ParamResolver,
)


class TestParamResolverStatic:
    def test_literals_become_static_params(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": 0.85, "sza": 60.0})
        static = r.static_params()
        assert static["albedo"] == (0.85, PROV_LITERAL)
        assert static["sza"] == (60.0, PROV_LITERAL)

    def test_unknown_key_is_unvalidated_passthrough(self, minimal_config):
        r = ParamResolver(minimal_config, {"crs_model": "rayleigh Bodhaine"})
        assert r.static_params()["crs_model"] == (
            "rayleigh Bodhaine",
            PROV_UNVALIDATED,
        )

    def test_literal_validation_raises_before_run(self, minimal_config):
        with pytest.raises(ValidationError):
            ParamResolver(minimal_config, {"albedo": 1.5})

    def test_validation_error_lists_all_offenders(self, minimal_config):
        with pytest.raises(ValidationError) as exc:
            ParamResolver(minimal_config, {"albedo": 1.5, "sza": 999.0})
        assert "albedo" in str(exc.value)
        assert "sza" in str(exc.value)

    def test_dotted_keys_applied_to_config_and_consumed(self, minimal_config):
        # B1 regression: dotted keys must never reach uvspec params
        r = ParamResolver(minimal_config, {"simulation_defaults.albedo_value": 0.3})
        assert minimal_config.simulation_defaults.albedo_value == 0.3
        assert "simulation_defaults.albedo_value" not in r.static_params()

    def test_unknown_dotted_key_warns_and_is_dropped(self, minimal_config, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            r = ParamResolver(minimal_config, {"simulation_defaults.nope": 1})
        assert "simulation_defaults.nope" not in r.static_params()
        assert "Unknown config parameter" in caplog.text

    def test_var_refs_separated(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": Var("alb_col")})
        assert r.var_refs == {"albedo": Var("alb_col")}
        assert "albedo" not in r.static_params()

    def test_none_params_ok(self, minimal_config):
        r = ParamResolver(minimal_config, None)
        assert r.static_params() == {}
        assert r.var_refs == {}


import numpy as np
import xarray as xr

from pyradtran.params import PROV_DATASET


def _point_ds(**values):
    """One stacked point: each kwarg becomes a 0-d variable."""
    return xr.Dataset({k: ((), v) for k, v in values.items()})


class TestResolvePoint:
    def test_var_resolved_from_point(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": Var("alb")})
        resolved, skipped = r.resolve_point(_point_ds(alb=0.7))
        assert resolved["albedo"] == (0.7, PROV_DATASET)
        assert skipped == []

    def test_nan_skips_parameter_and_records(self, minimal_config):
        # B4 regression: NaN must never reach the input file
        r = ParamResolver(minimal_config, {"albedo": Var("alb")})
        resolved, skipped = r.resolve_point(_point_ds(alb=np.nan))
        assert "albedo" not in resolved
        assert skipped == ["albedo"]

    def test_static_and_var_merge(self, minimal_config):
        r = ParamResolver(minimal_config, {"sza": 45.0, "albedo": Var("alb")})
        resolved, _ = r.resolve_point(_point_ds(alb=0.2))
        assert resolved["sza"] == (45.0, PROV_LITERAL)
        assert resolved["albedo"] == (0.2, PROV_DATASET)

    def test_resolved_var_value_is_validated(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": Var("alb")})
        with pytest.raises(ValidationError):
            r.resolve_point(_point_ds(alb=3.0))

    def test_validate_var_targets_missing(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": Var("nope")})
        ds = xr.Dataset({"alb": (("time",), [0.1])})
        with pytest.raises(ValidationError) as exc:
            r.validate_var_targets(ds)
        assert "nope" in str(exc.value)

    def test_bare_string_matching_dataset_var_raises(self, minimal_config):
        # Replaces the old silent string-hijack magic
        ds = xr.Dataset({"alb": (("time",), [0.1])})
        r = ParamResolver(minimal_config, {"albedo": "alb"})
        with pytest.raises(ValidationError) as exc:
            r.validate_var_targets(ds)
        assert "Var(" in str(exc.value)

    def test_missing_var_target_at_point_raises(self, minimal_config):
        r = ParamResolver(minimal_config, {"albedo": Var("absent")})
        with pytest.raises(ValidationError, match="absent"):
            r.resolve_point(_point_ds(other=1.0))


# append to tests/test_params.py
import warnings
from unittest.mock import patch

from pyradtran.interface import _translate_legacy_kwargs


class TestLegacyKwargTranslation:
    def test_var_kwargs_become_params(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = _translate_legacy_kwargs(
                params=None,
                albedo_var="alb",
                surface_temperature_var="skin_t",
                surface_type_var="igbp",
                altitude_var="flight_alt",
                parameter_overrides={"number_of_streams": 16},
            )
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert params["albedo"] == Var("alb")
        assert params["sur_temperature"] == Var("skin_t")
        assert params["brdf_rpv_type"] == Var("igbp")
        assert params["zout"] == Var("flight_alt")
        assert params["number_of_streams"] == 16

    def test_explicit_params_win_over_legacy(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            params = _translate_legacy_kwargs(
                params={"albedo": 0.5},
                albedo_var="alb",
                surface_temperature_var=None,
                surface_type_var=None,
                altitude_var=None,
                parameter_overrides=None,
            )
        assert params["albedo"] == 0.5

    def test_no_legacy_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = _translate_legacy_kwargs(
                params={"albedo": 0.5},
                albedo_var=None,
                surface_temperature_var=None,
                surface_type_var=None,
                altitude_var=None,
                parameter_overrides=None,
            )
        assert not any(issubclass(x.category, DeprecationWarning) for x in w)
        assert params == {"albedo": 0.5}


class TestBatchParams(object):
    def test_batch_resolves_var_per_point(self, minimal_config, simple_input_dataset):
        """Batch must hand per-point resolved params to the worker."""
        captured = []

        def fake_worker(config, point, _resolver_unused=None):
            captured.append(point)
            return None

        from pyradtran import interface

        ds = simple_input_dataset.copy()
        ds["alb"] = (["time"], [0.1, 0.2, 0.3])

        with patch.object(
            interface, "_run_single_simulation_unified", side_effect=fake_worker
        ):
            with pytest.raises(Exception):
                # All workers return None -> batch raises "all failed"
                interface.execute_simulation_batch(
                    config=minimal_config,
                    input_ds=ds,
                    params={"albedo": Var("alb")},
                    show_progress=False,
                )
        albs = sorted(p.resolved["albedo"][0] for p in captured)
        assert albs == [0.1, 0.2, 0.3]

    def test_missing_var_target_raises_before_submission(
        self, minimal_config, simple_input_dataset
    ):
        from pyradtran.interface import execute_simulation_batch

        with pytest.raises(ValidationError):
            execute_simulation_batch(
                config=minimal_config,
                input_ds=simple_input_dataset,
                params={"albedo": Var("does_not_exist")},
                show_progress=False,
            )

    def test_max_workers_none_uses_pool(self, minimal_config, simple_input_dataset):
        """max_workers=None must select the process pool (auto workers)."""
        from unittest.mock import patch

        from pyradtran import interface

        minimal_config.execution.max_workers = None
        with patch.object(interface, "ProcessPoolExecutor") as mock_pool:
            mock_pool.return_value.__enter__.return_value.submit.side_effect = (
                RuntimeError("stop")
            )
            with pytest.raises(Exception):
                interface.execute_simulation_batch(
                    config=minimal_config,
                    input_ds=simple_input_dataset,
                    show_progress=False,
                )
        mock_pool.assert_called_once_with(max_workers=None)


class TestPublicAPI:
    def test_top_level_exports(self):
        import pyradtran

        assert pyradtran.Var is Var
        assert callable(pyradtran.convolve_channels)
        assert callable(pyradtran.brightness_temperature)
        assert pyradtran.__version__ == "0.2.0"


# ---------------------------------------------------------------------------
# Schema-backed validation (libRadtran option schema)
# ---------------------------------------------------------------------------


@pytest.fixture
def mini_schema():
    """Hand-built schema mirroring the real extraction format."""
    return {
        "albedo": {
            "name": "albedo",
            "group": "Surface",
            "help": "",
            "doc": "",
            "non_unique": False,
            "mandatory": False,
            "parents": [],
            "childs": [],
            "tokens": [
                {
                    "kind": "value",
                    "datatype": "float",
                    "valid_range": [0.0, 1.0],
                    "optional": False,
                }
            ],
        },
        "ic_properties": {
            "name": "ic_properties",
            "group": "Clouds",
            "help": "",
            "doc": "",
            "non_unique": False,
            "mandatory": False,
            "parents": ["ic_file"],
            "childs": [],
            "tokens": [
                {
                    "kind": "choice",
                    "choices": ["fu", "baum_v36", "yang"],
                    "file_allowed": False,
                    "optional": False,
                },
                {
                    "kind": "choice",
                    "choices": ["interpolate"],
                    "file_allowed": False,
                    "optional": True,
                },
            ],
        },
        "wc_file": {
            "name": "wc_file",
            "group": "Clouds",
            "help": "",
            "doc": "",
            "non_unique": False,
            "mandatory": False,
            "parents": [],
            "childs": [],
            "tokens": [
                {
                    "kind": "choice",
                    "choices": ["1d", "3d", "ipa_files"],
                    "file_allowed": False,
                    "optional": False,
                },
                {
                    "kind": "value",
                    "datatype": "file",
                    "valid_range": None,
                    "optional": False,
                },
            ],
        },
        "cloudcover": {
            "name": "cloudcover",
            "group": "Clouds",
            "help": "",
            "doc": "",
            "non_unique": True,
            "mandatory": False,
            "parents": ["ic_file", "wc_file"],
            "childs": [],
            "tokens": [
                {
                    "kind": "value",
                    "datatype": "str",
                    "valid_range": None,
                    "optional": False,
                },
                {
                    "kind": "value",
                    "datatype": "float",
                    "valid_range": [0.0, 1.0],
                    "optional": False,
                },
            ],
        },
        "zout": {
            "name": "zout",
            "group": "Output",
            "help": "",
            "doc": "",
            "non_unique": False,
            "mandatory": False,
            "parents": [],
            "childs": [],
            "tokens": [
                {
                    "kind": "value",
                    "datatype": "str",
                    "valid_range": None,
                    "optional": False,
                }
            ],
        },
        "umu": {
            "name": "umu",
            "group": "Geometry",
            "help": "",
            "doc": "",
            "non_unique": False,
            "mandatory": False,
            "parents": [],
            "childs": [],
            "tokens": [
                {
                    "kind": "value",
                    "datatype": "floats",
                    "valid_range": None,
                    "optional": False,
                }
            ],
        },
        "quiet": {
            "name": "quiet",
            "group": "Output",
            "help": "",
            "doc": "",
            "non_unique": False,
            "mandatory": False,
            "parents": [],
            "childs": [],
            "tokens": [],
        },
    }


class TestSchemaValidation:
    def test_unknown_option_rejected_with_suggestion(self, minimal_config, mini_schema):
        with pytest.raises(ValidationError, match="albedo"):
            ParamResolver(minimal_config, {"albedoo": 0.5}, schema=mini_schema)

    def test_raw_bypasses_validation(self, minimal_config, mini_schema):
        from pyradtran.params import PROV_UNVALIDATED, Raw

        r = ParamResolver(minimal_config, {"albedoo": Raw("0.5")}, schema=mini_schema)
        assert r.static_params()["albedoo"] == ("0.5", PROV_UNVALIDATED)

    def test_choice_validated(self, minimal_config, mini_schema):
        ParamResolver(minimal_config, {"ic_properties": "baum_v36"}, schema=mini_schema)
        with pytest.raises(ValidationError, match="ic_properties"):
            ParamResolver(
                minimal_config, {"ic_properties": "notahabit"}, schema=mini_schema
            )

    def test_choice_case_insensitive(self, minimal_config, mini_schema, tmp_path):
        # users write "1D", the schema stores "1d"
        ParamResolver(
            minimal_config,
            {"wc_file": f"1D {tmp_path}/wc.dat"},
            schema=mini_schema,
        )

    def test_multiword_key_tokens_count(self, minimal_config, mini_schema, tmp_path):
        # key carries the first token, value the second
        ParamResolver(
            minimal_config,
            {"wc_file 1D": f"{tmp_path}/wc.dat"},
            schema=mini_schema,
        )

    def test_range_validated(self, minimal_config, mini_schema):
        ParamResolver(minimal_config, {"cloudcover": ("wc", 0.8)}, schema=mini_schema)
        with pytest.raises(ValidationError, match="cloudcover"):
            ParamResolver(
                minimal_config, {"cloudcover": ("wc", 1.5)}, schema=mini_schema
            )

    def test_greedy_str_token_accepts_many(self, minimal_config, mini_schema):
        ParamResolver(minimal_config, {"zout": "0.0 1.0 120.0"}, schema=mini_schema)

    def test_floats_token_rejects_nonnumeric(self, minimal_config, mini_schema):
        ParamResolver(minimal_config, {"umu": "-1.0 1.0"}, schema=mini_schema)
        with pytest.raises(ValidationError, match="umu"):
            ParamResolver(minimal_config, {"umu": "-1.0 up"}, schema=mini_schema)

    def test_flag_option_takes_no_value(self, minimal_config, mini_schema):
        ParamResolver(minimal_config, {"quiet": True}, schema=mini_schema)
        with pytest.raises(ValidationError, match="quiet"):
            ParamResolver(minimal_config, {"quiet": "loudly"}, schema=mini_schema)

    def test_fixed_arity_rejects_extra_tokens(self, minimal_config, mini_schema):
        with pytest.raises(ValidationError, match="albedo"):
            ParamResolver(minimal_config, {"albedo": "0.5 0.6"}, schema=mini_schema)

    def test_dict_value_skips_schema(self, minimal_config, mini_schema):
        # dynamic-cloud dict values are handled downstream
        ParamResolver(
            minimal_config,
            {"wc_file": {"z": [2, 1], "lwc": [0.1, 0.1], "reff": [10, 10]}},
            schema=mini_schema,
        )

    def test_curated_registry_wins_over_schema(self, minimal_config, mini_schema):
        # albedo stays validated by the curated ParamSpec ([0,1])
        with pytest.raises(ValidationError):
            ParamResolver(minimal_config, {"albedo": 1.5}, schema=mini_schema)

    def test_no_schema_keeps_passthrough(self, minimal_config):
        from pyradtran.params import PROV_UNVALIDATED

        r = ParamResolver(minimal_config, {"whatever_odd_key": "x"}, schema=None)
        assert r.static_params()["whatever_odd_key"] == ("x", PROV_UNVALIDATED)

    def test_var_value_schema_validated_per_point(self, minimal_config, mini_schema):
        import xarray as xr

        r = ParamResolver(
            minimal_config,
            {"ic_properties": Var("habit")},
            schema=mini_schema,
        )
        good = xr.Dataset({"habit": ((), "fu")})
        resolved, _ = r.resolve_point(good)
        assert resolved["ic_properties"][0] == "fu"
        bad = xr.Dataset({"habit": ((), "granite")})
        with pytest.raises(ValidationError):
            r.resolve_point(bad)


class TestValueNormalisation:
    def test_tuple_value_normalised_to_string(self, minimal_config, mini_schema):
        r = ParamResolver(
            minimal_config, {"cloudcover": ("wc", 0.8)}, schema=mini_schema
        )
        assert r.static_params()["cloudcover"][0] == "wc 0.8"

    def test_list_on_unique_option_joined(self, minimal_config, mini_schema):
        r = ParamResolver(
            minimal_config, {"zout": [0.0, 1.0, 120.0]}, schema=mini_schema
        )
        assert r.static_params()["zout"][0] == "0.0 1.0 120.0"

    def test_list_on_non_unique_option_kept_as_lines(self, minimal_config):
        schema = {
            "wc_modify": {
                "name": "wc_modify",
                "group": "Clouds",
                "help": "",
                "doc": "",
                "non_unique": True,
                "mandatory": False,
                "parents": ["wc_file"],
                "childs": [],
                "tokens": [
                    {
                        "kind": "choice",
                        "choices": ["gg", "ssa", "tau", "tau550"],
                        "file_allowed": False,
                        "optional": False,
                    },
                    {
                        "kind": "choice",
                        "choices": ["set", "scale"],
                        "file_allowed": False,
                        "optional": False,
                    },
                    {
                        "kind": "value",
                        "datatype": "float",
                        "valid_range": None,
                        "optional": False,
                    },
                ],
            },
        }
        r = ParamResolver(
            minimal_config,
            {"wc_modify": [("tau550", "set", 12), "ssa set 0.99"]},
            schema=schema,
        )
        assert r.static_params()["wc_modify"][0] == ["tau550 set 12", "ssa set 0.99"]

    def test_each_line_of_non_unique_list_validated(self, minimal_config):
        schema = {
            "wc_modify": {
                "name": "wc_modify",
                "group": "Clouds",
                "help": "",
                "doc": "",
                "non_unique": True,
                "mandatory": False,
                "parents": [],
                "childs": [],
                "tokens": [
                    {
                        "kind": "choice",
                        "choices": ["gg", "ssa", "tau", "tau550"],
                        "file_allowed": False,
                        "optional": False,
                    },
                    {
                        "kind": "choice",
                        "choices": ["set", "scale"],
                        "file_allowed": False,
                        "optional": False,
                    },
                    {
                        "kind": "value",
                        "datatype": "float",
                        "valid_range": None,
                        "optional": False,
                    },
                ],
            },
        }
        with pytest.raises(ValidationError, match="wc_modify"):
            ParamResolver(
                minimal_config,
                {"wc_modify": ["tau550 set 12", "granite set 1"]},
                schema=schema,
            )


class TestDescribe:
    def test_describe_signature_and_help(self, mini_schema):
        from pyradtran.params import describe

        txt = describe("ic_properties", schema=mini_schema)
        assert "ic_properties" in txt
        assert "fu|baum_v36|yang" in txt
        assert "[interpolate]" in txt  # optional token bracketed

    def test_describe_cleans_latex(self):
        from pyradtran.params import describe

        schema = {
            "albedo": {
                "name": "albedo",
                "group": "Surface",
                "help": "Lambertian albedo.",
                "doc": r"Set the \code{albedo} of the surface, see \file{ALB.DAT}.",
                "non_unique": False,
                "mandatory": False,
                "parents": [],
                "childs": [],
                "tokens": [
                    {
                        "kind": "value",
                        "datatype": "float",
                        "valid_range": [0.0, 1.0],
                        "optional": False,
                    }
                ],
            }
        }
        txt = describe("albedo", schema=schema)
        assert "\\code" not in txt
        assert "`albedo`" in txt
        assert "ALB.DAT" in txt
        assert "[0.0, 1.0]" in txt

    def test_describe_unknown_raises(self, mini_schema):
        from pyradtran.params import describe

        with pytest.raises(KeyError):
            describe("nonexistent_thing", schema=mini_schema)

    def test_search_options(self, mini_schema):
        from pyradtran.params import search_options

        hits = search_options("cloud", schema=mini_schema)
        assert "ic_properties" in hits  # group "Clouds"
        assert "cloudcover" in hits
        assert "albedo" not in hits
