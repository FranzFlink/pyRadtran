# tests/test_config.py
"""
Tests for the configuration module
"""
import os
import tempfile
import pytest
from pathlib import Path
import yaml

from pyradtran.config import (
    PathsConfig, 
    SimulationDefaults,
    ExecutionConfig,
    OutputConfig,
    SimulationConfig,
    load_config,
    CloudParameters,
    AerosolParameters
)

# Test fixture for creating a temporary config file
@pytest.fixture
def temp_config_file():
    """Creates a temporary config file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
        # Write a minimal valid config
        config = {
            'paths': {
                'libradtran_bin': '/path/to/uvspec',
                'libradtran_data': '/path/to/data',
                'atmosphere_profile': '/path/to/atmosphere.dat',
                'solar_spectrum': '/path/to/solar.dat',
            },
            'simulation_defaults': {
                'rte_solver': 'disort',
                'wavelength_nm': [300, 800],
            },
            'execution': {
                'max_workers': 2,
            },
            'output': {
                'filename_prefix': 'test',
            }
        }
        
        yaml.dump(config, tmp)
        tmp_path = tmp.name
    
    yield tmp_path
    # Clean up
    os.unlink(tmp_path)

# Tests
def test_cloud_parameters_validation():
    """Test validation of cloud parameters"""
    # Valid configuration
    cp = CloudParameters(
        enabled=True,
        layer_heights_km=[(1.0, 2.0)],
        layer_water_content=[0.1],
        layer_effective_radius_um=[10.0]
    )
    
    # Should not raise errors
    assert cp.enabled is True
    assert len(cp.layer_heights_km) == 1
    
    # Test inconsistent layers
    with pytest.raises(ValueError):
        CloudParameters(
            enabled=True,
            layer_heights_km=[(1.0, 2.0), (3.0, 4.0)],
            layer_water_content=[0.1],  # Only one value
            layer_effective_radius_um=[10.0, 15.0]
        )

def test_aerosol_parameters():
    """Test aerosol parameters"""
    ap = AerosolParameters(
        enabled=True,
        aerosol_type="rural",
        aerosol_visibility_km=23.0
    )
    
    assert ap.enabled is True
    assert ap.aerosol_type == "rural"
    assert ap.aerosol_visibility_km == 23.0

def test_simulation_defaults_validation():
    """Test validation for simulation defaults"""
    # Valid configuration
    sd = SimulationDefaults(
        wavelength_nm=[400, 700],
        output_altitudes_km=[0.0, 1.0, 2.0]
    )
    
    assert sd.wavelength_nm == [400, 700]
    assert sd.output_altitudes_km == [0.0, 1.0, 2.0]
    
    # Test wavelength validation
    with pytest.raises(ValueError):
        SimulationDefaults(wavelength_nm=[400])  # Missing max value
    
    # Test altitude validation
    with pytest.raises(ValueError):
        SimulationDefaults(output_altitudes_km=[])  # Empty list

def test_simulation_config_from_yaml(temp_config_file, monkeypatch):
    """Test loading config from YAML file"""
    # Mock the file check to avoid actual file existence check
    def mock_is_file(self):
        return True
    
    def mock_is_dir(self):
        return True
    
    monkeypatch.setattr(Path, "is_file", mock_is_file)
    monkeypatch.setattr(Path, "is_dir", mock_is_dir)
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    
    # Test loading
    config = SimulationConfig.from_yaml(temp_config_file)
    
    assert isinstance(config, SimulationConfig)
    assert config.simulation_defaults.rte_solver == 'disort'
    assert config.execution.max_workers == 2
    assert config.output.filename_prefix == 'test'