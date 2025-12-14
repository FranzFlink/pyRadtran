import pytest
from pathlib import Path
import yaml
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock
from pyradtran.config import PathsConfig, load_config, SimulationConfig, _recursive_update

@pytest.fixture
def mock_master_config_path(tmp_path):
    """Fixture to mock the master config path."""
    config_dir = tmp_path / ".pyradtran"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    return config_file

def test_recursive_update():
    """Test recursive update of dictionaries."""
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    update = {'b': {'c': 4}, 'e': 5}
    expected = {'a': 1, 'b': {'c': 4, 'd': 3}, 'e': 5}
    
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
        solar_spectrum=data_dir / "custom_solar.dat"
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
    assert config_inferred.solar_spectrum == data_dir / "solar_flux" / "kurudz_1.0nm.dat"

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
        'paths': {
            'libradtran_bin': str(bin_path),
            'libradtran_data': str(data_path)
        },
        'execution': {
            'max_workers': 8
        }
    }
    with open(mock_master_config_path, 'w') as f:
        yaml.dump(master_config_content, f)

    # 2. Create Specific Config (Minimal)
    specific_config_content = {
        'simulation_defaults': {
            'rte_solver': 'mystic'
        },
        'execution': {
            'max_workers': 1  # Override master
        }
    }
    specific_config_file = tmp_path / "sim.yaml"
    with open(specific_config_file, 'w') as f:
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
    assert loaded_config.simulation_defaults.rte_solver == 'mystic'
    assert loaded_config.execution.max_workers == 1  # Specific overrides master
