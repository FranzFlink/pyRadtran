import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pyradtran.config import (
    ATMOSPHERE_PROFILES,
    SOLAR_SPECTRA,
    ExecutionConfig,
    OutputConfig,
    PathsConfig,
    SimulationConfig,
    SimulationDefaults,
    _recursive_update,
    _resolve_libradtran_shortname,
    list_atmosphere_profiles,
    list_solar_spectra,
    load_config,
    save_master_config,
)


@pytest.fixture
def mock_master_config_path(tmp_path):
    """Fixture to mock the master config path."""
    config_dir = tmp_path / ".pyradtran"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    return config_file


def test_recursive_update():
    """Test recursive update of dictionaries."""
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    update = {"b": {"c": 4}, "e": 5}
    expected = {"a": 1, "b": {"c": 4, "d": 3}, "e": 5}

    result = _recursive_update(base, update)
    assert result == expected


def test_path_inference(tmp_path):
    """Test standard path inference."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bin_file = tmp_path / "uvspec"
    bin_file.touch()

    # Create expected default files mocked
    (data_dir / "atmmod").mkdir()
    (data_dir / "atmmod" / "afglus.dat").touch()
    (data_dir / "solar_flux").mkdir()
    (data_dir / "solar_flux" / "kurudz_1.0nm.dat").touch()
    (data_dir / "custom.dat").touch()
    (data_dir / "custom_solar.dat").touch()

    # Case 1: All paths provided
    config = PathsConfig(
        libradtran_bin=bin_file,
        libradtran_data=data_dir,
        atmosphere_profile=data_dir / "custom.dat",
        solar_spectrum=data_dir / "custom_solar.dat",
    )
    # Mock existence manually for custom files as they are checked in post_init
    # We can just skip actual file check by mocking Path.is_file/is_dir or ensuring they exist
    # Here we won't mock, so we expect errors if they don't exist.
    # PathsConfig checks existence in post_init.

    # Case 2: Inference
    config_inferred = PathsConfig(
        libradtran_bin=bin_file,
        libradtran_data=data_dir,
        # atmosphere_profile and solar_spectrum omitted
    )
    assert config_inferred.atmosphere_profile == data_dir / "atmmod" / "afglus.dat"
    assert (
        config_inferred.solar_spectrum == data_dir / "solar_flux" / "kurudz_1.0nm.dat"
    )


def test_load_config_with_master(tmp_path, mock_master_config_path):
    """Test loading config combining master and specific config."""

    # Define paths
    bin_path = tmp_path / "uvspec"
    data_path = tmp_path / "data"
    bin_path.touch()
    data_path.mkdir()
    (data_path / "atmmod").mkdir()
    (data_path / "atmmod" / "afglus.dat").touch()
    (data_path / "solar_flux").mkdir()
    (data_path / "solar_flux" / "kurudz_1.0nm.dat").touch()

    # 1. Create Master Config
    master_config_content = {
        "paths": {
            "libradtran_bin": str(bin_path),
            "libradtran_data": str(data_path),
            "atmosphere_profile": str(data_path / "atmmod" / "afglus.dat"),
            "solar_spectrum": str(data_path / "solar_flux" / "kurudz_1.0nm.dat"),
        },
        "execution": {"max_workers": 8},
    }
    with open(mock_master_config_path, "w") as f:
        yaml.dump(master_config_content, f)

    # 2. Create Specific Config (Minimal)
    specific_config_content = {
        "simulation_defaults": {"rte_solver": "mystic"},
        "execution": {"max_workers": 1},  # Override master
    }
    specific_config_file = tmp_path / "sim.yaml"
    with open(specific_config_file, "w") as f:
        yaml.dump(specific_config_content, f)

    # 3. Patch Path.home to return tmp_path so it finds our mock master config
    with patch("pathlib.Path.home", return_value=tmp_path):
        loaded_config = load_config(specific_config_file)

    # Verify
    assert loaded_config.paths.libradtran_bin == bin_path
    assert loaded_config.paths.libradtran_data == data_path
    # Inferred paths should be set
    assert loaded_config.paths.atmosphere_profile == data_path / "atmmod" / "afglus.dat"

    # Check overrides
    assert loaded_config.simulation_defaults.rte_solver == "mystic"
    assert loaded_config.execution.max_workers == 1  # Specific overrides master


# ──────────────────────────────────────────────────────────────────────────────
# Tests for SimulationConfig.to_dict() and SimulationConfig.to_yaml()
# ──────────────────────────────────────────────────────────────────────────────


def _make_minimal_config(tmp_path):
    """Helper: build a valid SimulationConfig with tmp-path stubs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bin_file = tmp_path / "uvspec"
    bin_file.touch()
    (data_dir / "atmmod").mkdir()
    (data_dir / "atmmod" / "afglus.dat").touch()
    (data_dir / "solar_flux").mkdir()
    (data_dir / "solar_flux" / "kurudz_1.0nm.dat").touch()

    paths = PathsConfig(
        libradtran_bin=bin_file,
        libradtran_data=data_dir,
        output_dir=tmp_path / "out",
        working_dir=tmp_path / "work",
    )
    return SimulationConfig(
        paths=paths,
        simulation_defaults=SimulationDefaults(),
        execution=ExecutionConfig(max_workers=2),
        output=OutputConfig(),
    )


def test_to_dict_returns_string_paths(tmp_path):
    """to_dict() must convert all Path objects to plain strings."""
    cfg = _make_minimal_config(tmp_path)
    d = cfg.to_dict()

    assert isinstance(d["paths"]["libradtran_bin"], str)
    assert isinstance(d["paths"]["libradtran_data"], str)
    assert isinstance(d["paths"]["output_dir"], str)
    assert isinstance(d["paths"]["working_dir"], str)


def test_to_dict_simulation_defaults(tmp_path):
    """to_dict() must preserve simulation_defaults values."""
    cfg = _make_minimal_config(tmp_path)
    cfg.simulation_defaults.rte_solver = "disort"
    cfg.simulation_defaults.wavelength_nm = [300, 900]

    d = cfg.to_dict()
    assert d["simulation_defaults"]["rte_solver"] == "disort"
    assert d["simulation_defaults"]["wavelength_nm"] == [300, 900]


def test_to_dict_no_era5_dataset_key(tmp_path):
    """to_dict() must not include the non-serialisable era5_dataset key."""
    cfg = _make_minimal_config(tmp_path)
    d = cfg.to_dict()
    assert "era5_dataset" not in d["simulation_defaults"]["clouds"]


def test_to_yaml_creates_file(tmp_path):
    """to_yaml() must write a YAML file that can be read back."""
    cfg = _make_minimal_config(tmp_path)
    out = tmp_path / "subdir" / "sim.yaml"

    result = cfg.to_yaml(out)

    assert result == out
    assert out.is_file()
    content = yaml.safe_load(out.read_text())
    assert "paths" in content
    assert "simulation_defaults" in content


def test_to_yaml_roundtrip(tmp_path):
    """Config serialised via to_yaml() then loaded via load_config() should
    reproduce the same key settings."""
    cfg = _make_minimal_config(tmp_path)
    cfg.simulation_defaults.rte_solver = "twostr"
    cfg.simulation_defaults.albedo_value = 0.42
    cfg.execution.max_workers = 3

    out_yaml = tmp_path / "roundtrip.yaml"
    cfg.to_yaml(out_yaml)

    loaded = load_config(out_yaml)

    assert loaded.simulation_defaults.rte_solver == "twostr"
    assert loaded.simulation_defaults.albedo_value == pytest.approx(0.42)
    assert loaded.execution.max_workers == 3


# ──────────────────────────────────────────────────────────────────────────────
# Tests for save_master_config()
# ──────────────────────────────────────────────────────────────────────────────


def test_save_master_config_creates_file(tmp_path):
    """save_master_config() must write ~/.pyradtran/config.yaml."""
    with patch("pathlib.Path.home", return_value=tmp_path):
        result = save_master_config(
            libradtran_bin="/opt/libradtran/bin/uvspec",
            libradtran_data="/opt/libradtran/data",
        )

    expected = tmp_path / ".pyradtran" / "config.yaml"
    assert result == expected
    assert expected.is_file()


def test_save_master_config_content(tmp_path):
    """save_master_config() must write correct paths to the YAML."""
    with patch("pathlib.Path.home", return_value=tmp_path):
        save_master_config(
            libradtran_bin="/opt/bin/uvspec",
            libradtran_data="/opt/data",
            radiosonde_base="/data/radiosondes",
            max_workers=4,
        )

    master = tmp_path / ".pyradtran" / "config.yaml"
    content = yaml.safe_load(master.read_text())

    assert content["paths"]["libradtran_bin"] == "/opt/bin/uvspec"
    assert content["paths"]["libradtran_data"] == "/opt/data"
    assert content["paths"]["radiosonde_base"] == "/data/radiosondes"
    assert content["execution"]["max_workers"] == 4


def test_save_master_config_extra(tmp_path):
    """save_master_config() extra kwarg must be merged into the file."""
    with patch("pathlib.Path.home", return_value=tmp_path):
        save_master_config(
            libradtran_bin="/opt/bin/uvspec",
            libradtran_data="/opt/data",
            extra={"execution": {"debug_mode": True, "timeout_seconds": 120}},
        )

    master = tmp_path / ".pyradtran" / "config.yaml"
    content = yaml.safe_load(master.read_text())
    assert content["execution"]["debug_mode"] is True
    assert content["execution"]["timeout_seconds"] == 120


def test_load_config_no_path_uses_master(tmp_path):
    """load_config() with no path argument must honour master config values."""
    # Set up minimal master config
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bin_file = tmp_path / "uvspec"
    bin_file.touch()
    (data_dir / "atmmod").mkdir()
    atm_file = data_dir / "atmmod" / "afglus.dat"
    atm_file.touch()
    (data_dir / "solar_flux").mkdir()
    solar_file = data_dir / "solar_flux" / "kurudz_1.0nm.dat"
    solar_file.touch()

    master_content = {
        "paths": {
            "libradtran_bin": str(bin_file),
            "libradtran_data": str(data_dir),
            "atmosphere_profile": str(atm_file),
            "solar_spectrum": str(solar_file),
        },
        "simulation_defaults": {
            "rte_solver": "rodents",
        },
    }
    master_dir = tmp_path / ".pyradtran"
    master_dir.mkdir()
    (master_dir / "config.yaml").write_text(yaml.dump(master_content))

    with patch("pathlib.Path.home", return_value=tmp_path):
        cfg = load_config()  # no explicit config_path

    assert cfg.simulation_defaults.rte_solver == "rodents"
    assert cfg.paths.libradtran_bin == bin_file


# ---------------------------------------------------------------------------
# Short-name resolution tests
# ---------------------------------------------------------------------------


def test_catalogs_not_empty():
    """SOLAR_SPECTRA and ATMOSPHERE_PROFILES contain at least the basic entries."""
    assert "kurudz_1.0nm" in SOLAR_SPECTRA
    assert "kurudz_0.1nm" in SOLAR_SPECTRA
    assert "afglus" in ATMOSPHERE_PROFILES
    assert "afglms" in ATMOSPHERE_PROFILES


def test_catalog_entries_are_tuples():
    """Every catalog entry is a (path_str, description) tuple."""
    for name, entry in SOLAR_SPECTRA.items():
        assert isinstance(entry, tuple) and len(entry) == 2, name
    for name, entry in ATMOSPHERE_PROFILES.items():
        assert isinstance(entry, tuple) and len(entry) == 2, name


def test_resolve_shortname_solar(tmp_path):
    """Short name resolves to data_root/relative_path."""
    result = _resolve_libradtran_shortname("kurudz_1.0nm", tmp_path, SOLAR_SPECTRA)
    assert result == tmp_path / "solar_flux" / "kurudz_1.0nm.dat"


def test_resolve_shortname_atmosphere(tmp_path):
    """Short name resolves to data_root/relative_path."""
    result = _resolve_libradtran_shortname("afglus", tmp_path, ATMOSPHERE_PROFILES)
    assert result == tmp_path / "atmmod" / "afglus.dat"


def test_resolve_shortname_none():
    """None input returns None."""
    result = _resolve_libradtran_shortname(None, Path("/data"), SOLAR_SPECTRA)
    assert result is None


def test_resolve_shortname_absolute_path(tmp_path):
    """An absolute path passes through unchanged."""
    p = tmp_path / "custom" / "my_solar.dat"
    result = _resolve_libradtran_shortname(str(p), Path("/data"), SOLAR_SPECTRA)
    assert result == p


def test_pathsconfig_short_name_solar(tmp_path):
    """PathsConfig accepts a short solar spectrum name."""
    bin_file = tmp_path / "uvspec"
    bin_file.touch()
    data_dir = tmp_path / "data"
    (data_dir / "solar_flux").mkdir(parents=True)
    (data_dir / "atmmod").mkdir(parents=True)
    solar = data_dir / "solar_flux" / "kurudz_1.0nm.dat"
    solar.touch()
    atm = data_dir / "atmmod" / "afglus.dat"
    atm.touch()

    cfg = PathsConfig(
        libradtran_bin=bin_file,
        libradtran_data=data_dir,
        solar_spectrum="kurudz_1.0nm",  # short name
        atmosphere_profile="afglus",  # short name
    )
    assert cfg.solar_spectrum == solar
    assert cfg.atmosphere_profile == atm


def test_list_solar_spectra_prints(capsys):
    """list_solar_spectra() produces non-empty output."""
    list_solar_spectra()
    captured = capsys.readouterr()
    assert "kurudz_1.0nm" in captured.out


def test_list_atmosphere_profiles_prints(capsys):
    """list_atmosphere_profiles() produces non-empty output."""
    list_atmosphere_profiles()
    captured = capsys.readouterr()
    assert "afglus" in captured.out
