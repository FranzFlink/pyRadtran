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

    def test_brdf_suppresses_albedo_regardless_of_order(self, minimal_config):
        text, _ = build_text(
            minimal_config,
            resolved={
                "brdf_rpv_type": (7, PROV_LITERAL),
                "albedo": (0.3, PROV_DATASET),
            },
        )
        first_words = [ln.split()[0] for ln in text.splitlines()]
        assert "albedo" not in first_words
        assert "brdf_rpv_type" in first_words

    def test_unknown_key_passthrough(self, minimal_config):
        from pyradtran.params import PROV_UNVALIDATED

        text, _ = build_text(
            minimal_config,
            resolved={"crs_model": ("rayleigh Bodhaine", PROV_UNVALIDATED)},
        )
        assert "crs_model rayleigh Bodhaine" in text


class TestOutputColumns:
    """The output_user line and the parser must agree on columns.

    Spectral runs need a lambda column (and multi-altitude runs a zout
    column) to reconstruct the wavelength/altitude axes; without them
    the batch converter used to silently produce all-NaN results.
    """

    def _output_user_line(self, text):
        lines = [ln for ln in text.splitlines() if ln.startswith("output_user")]
        assert len(lines) == 1
        return lines[0]

    def test_spectral_injects_lambda(self, minimal_config):
        # minimal_config is spectral (integrate_wavelength=False) but its
        # default columns lack lambda
        minimal_config.simulation_defaults.output_columns = ["sza", "eglo"]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        text, _ = build_text(minimal_config)
        cols = self._output_user_line(text).split()[1:]
        assert "lambda" in cols
        assert cols[-2:] == ["sza", "eglo"]

    def test_integrated_does_not_inject_lambda(self, minimal_config):
        minimal_config.simulation_defaults.integrate_wavelength = True
        minimal_config.simulation_defaults.output_columns = ["sza", "eglo"]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        text, _ = build_text(minimal_config)
        assert self._output_user_line(text) == "output_user sza eglo"

    def test_multi_altitude_injects_zout(self, minimal_config):
        minimal_config.simulation_defaults.integrate_wavelength = True
        minimal_config.simulation_defaults.output_columns = ["eglo"]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0, 1.0]
        text, _ = build_text(minimal_config)
        cols = self._output_user_line(text).split()[1:]
        assert "zout" in cols

    def test_lambda_not_duplicated(self, minimal_config):
        minimal_config.simulation_defaults.output_columns = ["lambda", "eglo"]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        text, _ = build_text(minimal_config)
        assert self._output_user_line(text) == "output_user lambda eglo"

    def test_output_user_override_replaces_config_columns(self, minimal_config):
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        text, _ = build_text(
            minimal_config,
            resolved={"output_user": ("sza edir", PROV_LITERAL)},
        )
        line = self._output_user_line(text)
        cols = line.split()[1:]
        # override wins over config columns, injection still applies
        assert cols[-2:] == ["sza", "edir"]
        assert "lambda" in cols
        assert "eglo" not in cols

    def test_zout_override_drives_injection(self, minimal_config):
        minimal_config.simulation_defaults.integrate_wavelength = True
        minimal_config.simulation_defaults.output_columns = ["eglo"]
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        text, _ = build_text(
            minimal_config, resolved={"zout": ("0.0 1.0 2.0", PROV_LITERAL)}
        )
        cols = self._output_user_line(text).split()[1:]
        assert "zout" in cols


class TestZoutOrdering:
    """uvspec hard-errors on unsorted zout; every zout path must sort."""

    def test_config_zout_sorted_after_direct_assignment(self, minimal_config):
        # the accessor assigns dataset altitudes directly, bypassing the
        # dataclass __post_init__ sort
        minimal_config.simulation_defaults.output_altitudes_km = [10.0, 0.0, 10.0]
        text, _ = build_text(minimal_config)
        assert "zout 0.0000 10.0000" in text

    def test_zout_override_sorted_and_deduped(self, minimal_config):
        text, _ = build_text(
            minimal_config, resolved={"zout": ("10 0 10", PROV_LITERAL)}
        )
        zout_line = [l for l in text.splitlines() if l.startswith("zout")][0]
        assert zout_line == "zout 0 10"

    def test_zout_symbolic_tokens_pass_through(self, minimal_config):
        text, _ = build_text(
            minimal_config, resolved={"zout": ("toa", PROV_LITERAL)}
        )
        assert "zout toa" in text.splitlines()

    def test_effective_output_altitudes_sorted_dedup(self, minimal_config):
        from pyradtran.input_builder import effective_output_altitudes

        minimal_config.simulation_defaults.output_altitudes_km = [5.0, 0.0, 5.0]
        assert effective_output_altitudes(minimal_config) == [0.0, 5.0]
        assert effective_output_altitudes(minimal_config, {"zout": "3 1 3"}) == [
            1.0,
            3.0,
        ]
        assert effective_output_altitudes(minimal_config, {"zout": "toa"}) == ["toa"]


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


# append to tests/test_input_builder.py
import warnings


class TestSimulationWiring:
    def test_legacy_override_kwargs_warn_and_apply(self, minimal_config):
        from pyradtran.core import Simulation

        sim = Simulation(minimal_config)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lines = sim.build_input_lines(
                DT, 78.0, 15.0,
                resolved_params=None,
                override_albedo=0.9,
            )
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        text = "\n".join(ln.text for ln in lines)
        assert "albedo 0.9" in text

    def test_legacy_parameter_overrides_still_work(self, minimal_config):
        from pyradtran.core import Simulation

        sim = Simulation(minimal_config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            lines = sim.build_input_lines(
                DT, 78.0, 15.0, parameter_overrides={"number_of_streams": 16}
            )
        text = "\n".join(ln.text for ln in lines)
        assert "number_of_streams 16" in text

    def test_resolved_params_reach_input(self, minimal_config):
        from pyradtran.core import Simulation
        from pyradtran.params import PROV_LITERAL

        sim = Simulation(minimal_config)
        lines = sim.build_input_lines(
            DT, 78.0, 15.0, resolved_params={"albedo": (0.42, PROV_LITERAL)}
        )
        assert "albedo 0.42" in "\n".join(ln.text for ln in lines)

    def test_config_parameter_overrides_reach_input(self, minimal_config):
        from pyradtran.core import Simulation

        minimal_config.simulation_defaults.parameter_overrides = {
            "number_of_streams": 16
        }
        sim = Simulation(minimal_config)
        lines = sim.build_input_lines(DT, 78.0, 15.0)
        assert "number_of_streams 16" in "\n".join(ln.text for ln in lines)

    def test_runtime_parameter_overrides_beat_legacy_override(self, minimal_config):
        from pyradtran.core import Simulation

        sim = Simulation(minimal_config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            lines = sim.build_input_lines(
                DT, 78.0, 15.0,
                override_albedo=0.9,
                parameter_overrides={"albedo": 0.55},
            )
        text = "\n".join(ln.text for ln in lines)
        assert "albedo 0.55" in text
        assert "albedo 0.9" not in text

    def test_resolved_params_beat_everything(self, minimal_config):
        from pyradtran.core import Simulation
        from pyradtran.params import PROV_LITERAL

        minimal_config.simulation_defaults.parameter_overrides = {"albedo": 0.2}
        sim = Simulation(minimal_config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            lines = sim.build_input_lines(
                DT, 78.0, 15.0,
                resolved_params={"albedo": (0.42, PROV_LITERAL)},
                parameter_overrides={"albedo": 0.55},
            )
        text = "\n".join(ln.text for ln in lines)
        assert "albedo 0.42" in text


class TestValueRendering:
    def test_list_value_emits_multiple_lines(self, minimal_config):
        text, _ = build_text(
            minimal_config,
            resolved={"wc_modify": (["tau550 set 12", "ssa set 0.99"], PROV_LITERAL)},
        )
        lines = [ln for ln in text.splitlines() if ln.startswith("wc_modify")]
        assert lines == ["wc_modify tau550 set 12", "wc_modify ssa set 0.99"]

    def test_flag_value_true_renders_bare_keyword(self, minimal_config):
        text, _ = build_text(
            minimal_config, resolved={"aerosol_default": (True, PROV_LITERAL)}
        )
        assert "aerosol_default" in text.splitlines()
        assert "aerosol_default True" not in text


class TestRadiosondeColumns:
    """ERA5 atmosphere files: gas columns and mol_modify suppression."""

    def _era5_file(self, tmp_path, columns="H2O MMR O3 MMR"):
        f = tmp_path / "era5_atm.dat"
        f.write_text(
            "# ERA5 atmosphere profile (libRadtran radiosonde format)\n"
            f"# columns: {columns}\n"
            "# p(hPa)  T(K)  ...\n"
            "100.00  215.00  1.0000e-05  5.0000e-06\n"
            "1000.00  290.00  1.0000e-02  1.0000e-07\n"
        )
        return f

    def test_radiosonde_line_uses_header_columns(self, minimal_config, tmp_path):
        f = self._era5_file(tmp_path)
        text, _ = build_text(minimal_config, era5_atmosphere_file=f)
        assert f"radiosonde {f} H2O MMR O3 MMR" in text

    def test_o3_profile_suppresses_mol_modify_o3(self, minimal_config, tmp_path):
        minimal_config.simulation_defaults.ozone_du = 300.0
        f = self._era5_file(tmp_path)
        text, _ = build_text(minimal_config, era5_atmosphere_file=f)
        assert "mol_modify O3" not in text

    def test_h2o_profile_suppresses_mol_modify_h2o(self, minimal_config, tmp_path):
        minimal_config.simulation_defaults.h2o_mm = 2.0
        f = self._era5_file(tmp_path, columns="H2O MMR")
        text, _ = build_text(minimal_config, era5_atmosphere_file=f)
        assert "mol_modify H2O" not in text

    def test_no_o3_column_keeps_mol_modify_o3(self, minimal_config, tmp_path):
        minimal_config.simulation_defaults.ozone_du = 300.0
        f = self._era5_file(tmp_path, columns="H2O MMR")
        text, _ = build_text(minimal_config, era5_atmosphere_file=f)
        assert "mol_modify O3 300.0 DU" in text

    def test_legacy_file_without_columns_header(self, minimal_config, tmp_path):
        f = tmp_path / "legacy.dat"
        f.write_text(
            "# ERA5 atmosphere profile in libradtran radiosonde style\n"
            "# p(hPa)  T(K)  h2o(kg kg-1) \n"
            "100.00  215.00  1.000e-05\n"
        )
        text, _ = build_text(minimal_config, era5_atmosphere_file=f)
        assert f"radiosonde {f} H2O MMR" in text

    def test_legacy_rh_file_sniffed(self, minimal_config, tmp_path):
        f = tmp_path / "sonde.dat"
        f.write_text(
            "# Radiosonde atmosphere profile\n"
            "# p(hPa)  T(K)  h2o(RH%)\n"
            "100.00  215.00  30.0\n"
        )
        text, _ = build_text(minimal_config, era5_atmosphere_file=f)
        assert f"radiosonde {f} H2O RH" in text

    def test_no_atmosphere_file_keeps_mol_modify(self, minimal_config):
        minimal_config.simulation_defaults.ozone_du = 300.0
        minimal_config.simulation_defaults.h2o_mm = 2.0
        text, _ = build_text(minimal_config)
        assert "mol_modify O3 300.0 DU" in text
        assert "mol_modify H2O 2.0 MM" in text


class TestBrightnessColumns:
    """output_quantity brightness must drop albedo from output_user in the
    shared column function — builder and parser stay in lock-step even when
    albedo is not the last configured column."""

    def test_albedo_filtered_mid_list(self, minimal_config):
        from pyradtran.input_builder import effective_output_columns

        minimal_config.simulation_defaults.output_columns = [
            "sza", "albedo", "eglo",
        ]
        minimal_config.simulation_defaults.integrate_wavelength = True
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        cols = effective_output_columns(
            minimal_config, {"output_quantity": "brightness"}
        )
        assert cols == ["sza", "eglo"]

    def test_builder_output_user_line_matches(self, minimal_config):
        minimal_config.simulation_defaults.output_columns = [
            "sza", "albedo", "eglo",
        ]
        minimal_config.simulation_defaults.integrate_wavelength = True
        minimal_config.simulation_defaults.output_altitudes_km = [0.0]
        text, _ = build_text(
            minimal_config,
            resolved={"output_quantity": ("brightness", PROV_LITERAL)},
        )
        # albedo dropped from the requested output columns (the separate
        # surface-albedo option line is unaffected)
        out_lines = [
            l for l in text.splitlines() if l.startswith("output_user")
        ]
        assert out_lines == ["output_user sza eglo"]

    def test_parser_columns_match_builder(self, minimal_config):
        from pyradtran.input_builder import effective_output_columns
        from pyradtran.io import OutputParser

        minimal_config.simulation_defaults.output_columns = [
            "sza", "albedo", "eglo",
        ]
        minimal_config.simulation_defaults.integrate_wavelength = True
        overrides = {"output_quantity": "brightness"}
        parser = OutputParser(minimal_config, overrides)
        assert parser.output_columns == effective_output_columns(
            minimal_config, overrides
        )
        assert parser.is_brightness_output is True
