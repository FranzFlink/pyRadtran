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
        r = ParamResolver(
            minimal_config, {"simulation_defaults.albedo_value": 0.3}
        )
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
