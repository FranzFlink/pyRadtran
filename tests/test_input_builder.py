"""Tests for pyradtran/input_builder.py."""

from datetime import datetime

import pytest

from pyradtran.input_builder import (
    InputFileBuilder,
    InputLine,
    calculate_solar_zenith_angle,
)
from pyradtran.params import PROV_CONFIG, PROV_DATASET, PROV_LITERAL

DT = datetime(2022, 7, 1, 12, 0)


def build_text(config, resolved=None, **kwargs):
    b = InputFileBuilder(config)
    lines = b.build(DT, 78.0, 15.0, resolved=resolved, **kwargs)
    return b.render(lines), lines


class TestBuilderBasics:
    def test_config_albedo_emitted_with_config_provenance(self, minimal_config):
        text, lines = build_text(minimal_config)
        assert "albedo 0.1" in text
        alb = [ln for ln in lines if ln.keyword == "albedo"][0]
        assert alb.provenance == PROV_CONFIG

    def test_render_ends_with_newline_and_quiet(self, minimal_config):
        text, _ = build_text(minimal_config)
        assert text.endswith("\n")
        assert "quiet" in text.splitlines()

    def test_mol_modify_emitted_from_config(self, minimal_config):
        # B2 regression: ozone_du and h2o_mm must reach the input file
        minimal_config.simulation_defaults.ozone_du = 300.0
        minimal_config.simulation_defaults.h2o_mm = 2.0
        text, _ = build_text(minimal_config)
        assert "mol_modify O3 300.0 DU" in text
        assert "mol_modify H2O 2.0 MM" in text

    def test_mol_modify_not_emitted_when_none(self, minimal_config):
        minimal_config.simulation_defaults.ozone_du = None
        minimal_config.simulation_defaults.h2o_mm = None
        text, _ = build_text(minimal_config)
        assert not any(ln.startswith("mol_modify ") for ln in text.splitlines())

    def test_thermal_surface_temperature_zero_not_swallowed(self, minimal_config):
        # B3 regression: `or` swallowed falsy values
        minimal_config.simulation_defaults.source = "thermal"
        minimal_config.simulation_defaults.surface_temperature_k = 271.2
        text, _ = build_text(
            minimal_config, resolved={"sur_temperature": (0.0, PROV_DATASET)}
        )
        assert "sur_temperature 0.0" in text
        assert "sur_temperature 271.2" not in text


class TestOverrideReplacement:
    def test_resolved_replaces_config_line(self, minimal_config):
        text, lines = build_text(
            minimal_config, resolved={"albedo": (0.85, PROV_LITERAL)}
        )
        assert "albedo 0.85" in text
        assert "albedo 0.1" not in text
        alb = [ln for ln in lines if ln.keyword == "albedo"][0]
        assert alb.provenance == PROV_LITERAL

    def test_multiword_keyword_replacement(self, minimal_config):
        minimal_config.simulation_defaults.ozone_du = 300.0
        text, _ = build_text(
            minimal_config, resolved={"mol_modify O3": (350.0, PROV_LITERAL)}
        )
        assert "mol_modify O3 350.0 DU" in text
        assert "mol_modify O3 300.0 DU" not in text

    def test_multiword_replacement_does_not_clobber_sibling(self, minimal_config):
        minimal_config.simulation_defaults.ozone_du = 300.0
        minimal_config.simulation_defaults.h2o_mm = 2.0
        text, _ = build_text(
            minimal_config, resolved={"mol_modify O3": (350.0, PROV_LITERAL)}
        )
        assert "mol_modify H2O 2.0 MM" in text

    def test_brdf_type_disables_albedo(self, minimal_config):
        text, _ = build_text(
            minimal_config, resolved={"brdf_rpv_type": (7, PROV_DATASET)}
        )
        assert "brdf_rpv_library IGBP" in text
        assert "brdf_rpv_type 7" in text
        assert "albedo" not in [ln.split()[0] for ln in text.splitlines()]

    def test_unknown_key_passthrough(self, minimal_config):
        from pyradtran.params import PROV_UNVALIDATED

        text, _ = build_text(
            minimal_config,
            resolved={"crs_model": ("rayleigh Bodhaine", PROV_UNVALIDATED)},
        )
        assert "crs_model rayleigh Bodhaine" in text


class TestAnnotatedRender:
    def test_annotations_present(self, minimal_config):
        b = InputFileBuilder(minimal_config)
        lines = b.build(
            DT, 78.0, 15.0, resolved={"albedo": (0.85, PROV_LITERAL)}
        )
        annotated = b.render_annotated(lines)
        assert "# params-literal" in annotated
        assert "# config" in annotated


class TestSza:
    def test_solar_zenith_angle_reasonable(self):
        # Summer noon at 45N, 0E: sun high, SZA well below 45 deg
        sza = calculate_solar_zenith_angle(datetime(2022, 6, 21, 12, 0), 45.0, 0.0)
        assert 15.0 < sza < 30.0

    def test_resolved_sza_wins_over_calculated(self, minimal_config):
        text, _ = build_text(minimal_config, resolved={"sza": (77.7, PROV_LITERAL)})
        assert "sza 77.7" in text
