# tests/test_unified_system.py
"""
Comprehensive tests for the unified pyradtran system.

This test suite validates:
- Configuration loading and validation
- Core simulation functionality
- I/O operations (input loading, output parsing, NetCDF saving)
- Interface functions (batch execution, xarray accessor)
- ERA5 atmosphere file generation
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import unified components
from pyradtran.config_clean import SimulationConfig, load_config, create_example_config
from pyradtran.core_unified import Simulation
from pyradtran.io_unified import (
    InputDataLoader, ERA5AtmosphereGenerator, OutputParser, 
    OutputToXarray, NetCDFSaver, ParsedOutput, OutputType
)
from pyradtran.interface_unified import (
    run_pyradtran_simulation, execute_simulation_batch, PyRadtranAccessor
)
from pyradtran.exceptions import PyRadtranError, ConfigurationError, InputGenerationError


class TestConfigurationSystem:
    """Test the cleaned configuration system."""
    
    def test_load_example_config(self, tmp_path):
        """Test loading the example configuration."""
        config_path = tmp_path / "test_config.yaml"
        create_example_config(config_path)
        
        # Modify paths to valid dummy paths for testing
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create dummy files
        dummy_bin = tmp_path / "uvspec"
        dummy_bin.touch()
        dummy_bin.chmod(0o755)
        
        dummy_data = tmp_path / "data"
        dummy_data.mkdir()
        
        dummy_atm = tmp_path / "atmosphere.dat"
        dummy_atm.write_text("# dummy atmosphere\n0.0 1013.25 288.15 2.55e19 7.24e11 5.39e18 3.91e17 1.06e19 2.15e8\n")
        
        dummy_solar = tmp_path / "solar.dat"
        dummy_solar.write_text("# dummy solar spectrum\n280.0 4.09e-01\n")
        
        # Update config with valid paths
        config_data['paths']['libradtran_bin'] = str(dummy_bin)
        config_data['paths']['libradtran_data'] = str(dummy_data)
        config_data['paths']['atmosphere_profile'] = str(dummy_atm)
        config_data['paths']['solar_spectrum'] = str(dummy_solar)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Test loading
        config = load_config(config_path)
        
        assert isinstance(config, SimulationConfig)
        assert config.simulation_defaults.rte_solver == 'twostr'
        assert config.simulation_defaults.source == 'solar'
        assert config.simulation_defaults.wavelength_nm == [400, 3600]
        assert config.paths.libradtran_bin == dummy_bin
    
    def test_config_validation(self, tmp_path):
        """Test configuration validation."""
        config_path = tmp_path / "invalid_config.yaml"
        
        # Create config with invalid values
        invalid_config = {
            'paths': {
                'libradtran_bin': '/nonexistent/uvspec',
                'libradtran_data': '/nonexistent/data',
                'atmosphere_profile': '/nonexistent/atm.dat',
                'solar_spectrum': '/nonexistent/solar.dat'
            },
            'simulation_defaults': {
                'source': 'invalid_source',  # Should cause validation error
                'wavelength_nm': [400],  # Should be [min, max]
                'albedo_value': 1.5  # Should be 0-1
            }
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Should raise FileNotFoundError for missing files
        with pytest.raises(FileNotFoundError):
            load_config(config_path)
    
    def test_get_used_parameters(self, tmp_path):
        """Test getting used parameters from configuration."""
        config_path = tmp_path / "test_config.yaml"
        create_example_config(config_path)
        
        # Create dummy files (minimal setup)
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        dummy_files = ['uvspec', 'atmosphere.dat', 'solar.dat']
        for filename in dummy_files:
            dummy_file = tmp_path / filename
            dummy_file.touch()
            if filename == 'uvspec':
                dummy_file.chmod(0o755)
        
        dummy_data = tmp_path / "data"
        dummy_data.mkdir()
        
        config_data['paths']['libradtran_bin'] = str(tmp_path / 'uvspec')
        config_data['paths']['libradtran_data'] = str(dummy_data)
        config_data['paths']['atmosphere_profile'] = str(tmp_path / 'atmosphere.dat')
        config_data['paths']['solar_spectrum'] = str(tmp_path / 'solar.dat')
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(config_path)
        used_params = config.get_used_parameters()
        
        assert 'paths' in used_params
        assert 'simulation_defaults' in used_params
        assert 'execution' in used_params
        assert 'output' in used_params
        
        assert used_params['simulation_defaults']['rte_solver'] == 'twostr'
        assert used_params['simulation_defaults']['clouds']['enabled'] == False


class TestInputDataLoader:
    """Test input data loading functionality."""
    
    def test_load_csv_data(self, tmp_path):
        """Test loading CSV input data."""
        # Create test CSV
        csv_data = pd.DataFrame({
            'time': pd.date_range('2023-05-01', periods=3, freq='1H'),
            'latitude': [60.0, 60.1, 60.2],
            'longitude': [10.0, 10.1, 10.2],
            'albedo': [0.8, 0.85, 0.9]
        })
        
        csv_path = tmp_path / "test_input.csv"
        csv_data.to_csv(csv_path, index=False)
        
        # Load with InputDataLoader
        loader = InputDataLoader()
        ds = loader.load_simulation_input_data(csv_path)
        
        assert isinstance(ds, xr.Dataset)
        assert 'time' in ds.dims
        assert 'latitude' in ds.data_vars
        assert 'longitude' in ds.data_vars
        assert 'albedo' in ds.data_vars
        assert len(ds.time) == 3
    
    def test_load_netcdf_data(self, tmp_path):
        """Test loading NetCDF input data."""
        # Create test NetCDF
        times = pd.date_range('2023-05-01', periods=3, freq='1H')
        ds = xr.Dataset({
            'latitude': ('time', [60.0, 60.1, 60.2]),
            'longitude': ('time', [10.0, 10.1, 10.2]),
            'albedo': ('time', [0.8, 0.85, 0.9])
        }, coords={'time': times})
        
        nc_path = tmp_path / "test_input.nc"
        ds.to_netcdf(nc_path)
        
        # Load with InputDataLoader
        loader = InputDataLoader()
        loaded_ds = loader.load_simulation_input_data(nc_path)
        
        assert isinstance(loaded_ds, xr.Dataset)
        assert 'time' in loaded_ds.dims
        assert len(loaded_ds.time) == 3
    
    def test_load_invalid_data(self, tmp_path):
        """Test loading invalid input data."""
        # Create CSV without required columns
        csv_data = pd.DataFrame({
            'datetime': pd.date_range('2023-05-01', periods=3, freq='1H'),
            'lat': [60.0, 60.1, 60.2],  # Wrong column name
            'lon': [10.0, 10.1, 10.2]   # Wrong column name
        })
        
        csv_path = tmp_path / "invalid_input.csv"
        csv_data.to_csv(csv_path, index=False)
        
        loader = InputDataLoader()
        with pytest.raises(InputGenerationError):
            loader.load_simulation_input_data(csv_path)


class TestERA5AtmosphereGenerator:
    """Test ERA5 atmosphere file generation."""
    
    @pytest.fixture
    def mock_era5_dataset(self):
        """Create a mock ERA5 dataset."""
        pressure_levels = [1000, 850, 700, 500, 300, 200, 100]  # hPa
        
        ds = xr.Dataset({
            'z': (['pressure_level', 'latitude', 'longitude', 'valid_time'], 
                  np.random.random((len(pressure_levels), 1, 1, 1)) * 50000),  # Geopotential
            't': (['pressure_level', 'latitude', 'longitude', 'valid_time'], 
                  280 + np.random.random((len(pressure_levels), 1, 1, 1)) * 40),  # Temperature
            'q': (['pressure_level', 'latitude', 'longitude', 'valid_time'], 
                  np.random.random((len(pressure_levels), 1, 1, 1)) * 0.02),  # Specific humidity
            'o3': (['pressure_level', 'latitude', 'longitude', 'valid_time'], 
                   np.random.random((len(pressure_levels), 1, 1, 1)) * 1e-5)  # Ozone mass mixing ratio
        }, coords={
            'pressure_level': pressure_levels,
            'latitude': [60.0],
            'longitude': [10.0],
            'valid_time': [pd.Timestamp('2023-05-01T12:00:00')]
        })
        
        return ds
    
    def test_create_era5_atmosphere_file(self, tmp_path, mock_era5_dataset):
        """Test creating ERA5 atmosphere file."""
        output_path = tmp_path / "era5_atmosphere.dat"
        
        generator = ERA5AtmosphereGenerator()
        result_path = generator.create_era5_atmosphere_file(
            era5_ds=mock_era5_dataset,
            latitude=60.0,
            longitude=10.0,
            time='2023-05-01T12:00:00',
            output_filepath=output_path
        )
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Check file content
        content = output_path.read_text()
        lines = content.strip().split('\n')
        assert len(lines) >= 8  # Header + data lines
        assert '# ERA5 atmosphere profile' in lines[0]
        
        # Check that data lines have the expected number of columns
        data_lines = [line for line in lines if not line.startswith('#')]
        for line in data_lines:
            columns = line.split()
            assert len(columns) == 9  # z, p, T, air, o3, o2, h2o, co2, no2
    
    def test_era5_missing_variables(self):
        """Test ERA5 file generation with missing required variables."""
        # Create dataset missing required variables
        incomplete_ds = xr.Dataset({
            't': (['pressure_level'], [280, 270, 260]),  # Missing 'z', 'q', 'o3'
        }, coords={
            'pressure_level': [1000, 850, 700],
            'latitude': [60.0],
            'longitude': [10.0],
            'valid_time': [pd.Timestamp('2023-05-01T12:00:00')]
        })
        
        generator = ERA5AtmosphereGenerator()
        with pytest.raises(ValueError, match="Required variable"):
            generator.create_era5_atmosphere_file(
                era5_ds=incomplete_ds,
                latitude=60.0,
                longitude=10.0,
                time='2023-05-01T12:00:00',
                output_filepath='test.dat'
            )


class TestCoreSimulation:
    """Test core simulation functionality."""
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock configuration for testing."""
        # Create dummy files
        dummy_bin = tmp_path / "uvspec"
        dummy_bin.touch()
        dummy_bin.chmod(0o755)
        
        dummy_data = tmp_path / "data"
        dummy_data.mkdir()
        
        dummy_atm = tmp_path / "atmosphere.dat"
        dummy_atm.write_text("# dummy atmosphere\n")
        
        dummy_solar = tmp_path / "solar.dat"
        dummy_solar.write_text("# dummy solar spectrum\n")
        
        from pyradtran.config_clean import SimulationConfig, PathsConfig, SimulationDefaults
        
        return SimulationConfig(
            paths=PathsConfig(
                libradtran_bin=dummy_bin,
                libradtran_data=dummy_data,
                atmosphere_profile=dummy_atm,
                solar_spectrum=dummy_solar,
                output_dir=tmp_path / "output",
                working_dir=tmp_path / "work"
            ),
            simulation_defaults=SimulationDefaults()
        )
    
    def test_simulation_initialization(self, mock_config):
        """Test simulation initialization."""
        sim = Simulation(mock_config)
        assert sim.config == mock_config
        assert sim.radiosonde_finder is None  # No radiosonde base path
    
    def test_generate_input_content(self, mock_config):
        """Test input content generation."""
        sim = Simulation(mock_config)
        
        dt = datetime(2023, 5, 1, 12, 0, 0)
        content = sim._generate_input_content(
            dt=dt,
            latitude=60.0,
            longitude=10.0,
            override_albedo=0.9
        )
        
        assert isinstance(content, str)
        assert 'rte_solver twostr' in content
        assert 'mol_abs_param lowtran per_nm' in content
        assert 'albedo 0.9' in content
        assert 'sza' in content  # Solar zenith angle should be calculated
        assert 'day_of_year' in content
        assert 'wavelength 400 3600' in content
    
    def test_solar_zenith_angle_calculation(self, mock_config):
        """Test solar zenith angle calculation."""
        sim = Simulation(mock_config)
        
        # Test with known values
        dt = datetime(2023, 6, 21, 12, 0, 0)  # Summer solstice, noon
        sza = sim._calculate_solar_zenith_angle(dt, 0.0, 0.0)  # Equator
        
        assert isinstance(sza, float)
        assert 0.0 <= sza <= 90.0  # Should be reasonable for noon at solstice
    
    @patch('subprocess.run')
    def test_run_uvspec_success(self, mock_run, mock_config, tmp_path):
        """Test successful LibRadtran execution."""
        # Mock successful subprocess run
        mock_run.return_value = Mock(returncode=0, stderr="")
        
        sim = Simulation(mock_config)
        
        input_path = tmp_path / "test.inp"
        input_path.write_text("rte_solver twostr\n")
        
        output_path = tmp_path / "test.out"
        
        success = sim._run_uvspec(input_path, output_path)
        
        assert success == True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_run_uvspec_failure(self, mock_run, mock_config, tmp_path):
        """Test failed LibRadtran execution."""
        # Mock failed subprocess run
        mock_run.return_value = Mock(returncode=1, stderr="Error: invalid input")
        
        sim = Simulation(mock_config)
        
        input_path = tmp_path / "test.inp"
        input_path.write_text("invalid input\n")
        
        output_path = tmp_path / "test.out"
        
        success = sim._run_uvspec(input_path, output_path)
        
        assert success == False


class TestOutputParser:
    """Test output parsing functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for output parsing."""
        from pyradtran.config_clean import SimulationConfig, PathsConfig, SimulationDefaults
        
        return SimulationConfig(
            paths=PathsConfig(
                libradtran_bin=Path('/dummy/uvspec'),
                libradtran_data=Path('/dummy/data'),
                atmosphere_profile=Path('/dummy/atm.dat'),
                solar_spectrum=Path('/dummy/solar.dat')
            ),
            simulation_defaults=SimulationDefaults(
                output_columns=['sza', 'eglo', 'eup', 'albedo'],
                output_altitudes_km=[0.0]
            )
        )
    
    def test_parse_simple_output(self, tmp_path, mock_config):
        """Test parsing simple LibRadtran output."""
        # Create mock output file
        output_data = np.array([
            [0.0, 500.0, 30.0, 200.0, 50.0, 0.85],  # zout, lambda, sza, eglo, eup, albedo
            [0.0, 600.0, 30.0, 220.0, 55.0, 0.85],
            [0.0, 700.0, 30.0, 240.0, 60.0, 0.85]
        ])
        
        output_file = tmp_path / "test_output.out"
        np.savetxt(output_file, output_data)
        
        parser = OutputParser(mock_config)
        parsed = parser.parse_output_file(output_file)
        
        assert isinstance(parsed, ParsedOutput)
        assert parsed.output_type == OutputType.SPECTRAL_SINGLE_ALTITUDE
        assert 'sza' in parsed.data
        assert 'eglo' in parsed.data
        assert len(parsed.data['sza']) == 3
    
    def test_parse_empty_output(self, tmp_path, mock_config):
        """Test parsing empty output file."""
        output_file = tmp_path / "empty_output.out"
        output_file.write_text("")
        
        parser = OutputParser(mock_config)
        
        with pytest.raises(Exception):  # Should raise OutputParsingError
            parser.parse_output_file(output_file)


class TestXarrayAccessor:
    """Test xarray accessor functionality."""
    
    @pytest.fixture
    def test_dataset(self):
        """Create a test xarray dataset."""
        times = pd.date_range('2023-05-01', periods=3, freq='1H')
        return xr.Dataset({
            'latitude': ('time', [60.0, 60.1, 60.2]),
            'longitude': ('time', [10.0, 10.1, 10.2]),
            'albedo': ('time', [0.8, 0.85, 0.9])
        }, coords={'time': times})
    
    def test_accessor_registration(self, test_dataset):
        """Test that xarray accessor is properly registered."""
        assert hasattr(test_dataset, 'pyradtran')
        assert isinstance(test_dataset.pyradtran, PyRadtranAccessor)
    
    def test_accessor_validation(self, test_dataset):
        """Test input dataset validation in accessor."""
        accessor = test_dataset.pyradtran
        
        # Should not raise for valid dataset
        accessor._validate_input_dataset('time', 'latitude', 'longitude', 'albedo', None, None)
        
        # Should raise for missing variables
        with pytest.raises(PyRadtranError):
            accessor._validate_input_dataset('time', 'missing_lat', 'longitude', None, None, None)


def test_integration_example():
    """Test a simple integration example."""
    # This test shows how the unified system should work together
    try:
        # Create test data
        times = pd.date_range('2023-05-01T12:00:00', periods=2, freq='1H')
        test_data = pd.DataFrame({
            'time': times,
            'latitude': [60.0, 60.1],
            'longitude': [10.0, 10.1],
            'albedo': [0.8, 0.85]
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Save test data
            csv_path = tmp_path / "test_input.csv"
            test_data.to_csv(csv_path, index=False)
            
            # Create minimal config
            config_path = tmp_path / "config.yaml"
            create_example_config(config_path)
            
            # Update config with dummy paths (since we're not actually running LibRadtran)
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Create dummy files
            for filename in ['uvspec', 'atmosphere.dat', 'solar.dat']:
                dummy_file = tmp_path / filename
                dummy_file.touch()
                if filename == 'uvspec':
                    dummy_file.chmod(0o755)
            
            dummy_data = tmp_path / "data"
            dummy_data.mkdir()
            
            config_data['paths']['libradtran_bin'] = str(tmp_path / 'uvspec')
            config_data['paths']['libradtran_data'] = str(dummy_data)
            config_data['paths']['atmosphere_profile'] = str(tmp_path / 'atmosphere.dat')
            config_data['paths']['solar_spectrum'] = str(tmp_path / 'solar.dat')
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Test loading components
            config = load_config(config_path)
            assert isinstance(config, SimulationConfig)
            
            loader = InputDataLoader()
            input_ds = loader.load_simulation_input_data(csv_path)
            assert isinstance(input_ds, xr.Dataset)
            
            # Test simulation setup (without actually running LibRadtran)
            sim = Simulation(config)
            content = sim._generate_input_content(
                dt=datetime(2023, 5, 1, 12, 0, 0),
                latitude=60.0,
                longitude=10.0
            )
            assert isinstance(content, str)
            assert len(content) > 0
            
            print("Integration test passed - all components work together")
            
    except Exception as e:
        pytest.fail(f"Integration test failed: {str(e)}")


if __name__ == "__main__":
    # Run a simple test
    test_integration_example()
    print("All basic tests passed!")
