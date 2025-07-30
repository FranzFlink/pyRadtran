#!/usr/bin/env python3
"""
Validation script for the unified pyradtran system.

This script tests the refactored codebase to ensure all components work together.
"""

import sys
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime

def test_basic_imports():
    """Test that all unified components can be imported."""
    print("Testing basic imports...")
    
    try:
        from pyradtran.config_clean import SimulationConfig, load_config, create_example_config
        print("✓ Config system imported")
        
        from pyradtran.io_unified import InputDataLoader, ERA5AtmosphereGenerator, OutputParser
        print("✓ IO system imported")
        
        from pyradtran.core_unified import Simulation
        print("✓ Core system imported")
        
        from pyradtran.interface_unified import PyRadtranAccessor, run_pyradtran_simulation
        print("✓ Interface system imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_system():
    """Test the cleaned configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from pyradtran.config_clean import create_example_config, load_config
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create example config
            config_path = tmp_path / "test_config.yaml"
            create_example_config(config_path)
            print("✓ Example config created")
            
            # Modify config to have valid dummy paths
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
            dummy_atm.write_text("# dummy atmosphere\n0.0 1013.25 288.15\n")
            
            dummy_solar = tmp_path / "solar.dat"
            dummy_solar.write_text("# dummy solar\n400 1.0\n")
            
            # Update config
            config_data['paths']['libradtran_bin'] = str(dummy_bin)
            config_data['paths']['libradtran_data'] = str(dummy_data)
            config_data['paths']['atmosphere_profile'] = str(dummy_atm)
            config_data['paths']['solar_spectrum'] = str(dummy_solar)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Load config
            config = load_config(config_path)
            print("✓ Config loaded successfully")
            
            # Test used parameters
            used_params = config.get_used_parameters()
            print(f"✓ Found {len(used_params)} parameter sections")
            
            return True
            
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_io_system():
    """Test the unified IO system."""
    print("\nTesting IO system...")
    
    try:
        from pyradtran.io_unified import InputDataLoader
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test CSV
            test_data = pd.DataFrame({
                'time': pd.date_range('2023-05-01T12:00:00', periods=3, freq='1H'),
                'latitude': [60.0, 60.1, 60.2],
                'longitude': [10.0, 10.1, 10.2],
                'albedo': [0.8, 0.85, 0.9]
            })
            
            csv_path = tmp_path / "test_input.csv"
            test_data.to_csv(csv_path, index=False)
            
            # Load with InputDataLoader
            loader = InputDataLoader()
            ds = loader.load_simulation_input_data(csv_path)
            
            print(f"✓ Loaded dataset with {len(ds.time)} time points")
            print(f"✓ Dataset variables: {list(ds.data_vars.keys())}")
            
            return True
            
    except Exception as e:
        print(f"✗ IO test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_system():
    """Test the core simulation system."""
    print("\nTesting core system...")
    
    try:
        from pyradtran.core_unified import Simulation
        from pyradtran.config_clean import SimulationConfig, PathsConfig, SimulationDefaults
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create dummy files
            dummy_bin = tmp_path / "uvspec"
            dummy_bin.touch()
            dummy_bin.chmod(0o755)
            
            dummy_data = tmp_path / "data"
            dummy_data.mkdir()
            
            dummy_atm = tmp_path / "atmosphere.dat"
            dummy_atm.write_text("# dummy atmosphere\n")
            
            dummy_solar = tmp_path / "solar.dat"
            dummy_solar.write_text("# dummy solar\n")
            
            # Create config
            config = SimulationConfig(
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
            
            # Initialize simulation
            sim = Simulation(config)
            print("✓ Simulation initialized")
            
            # Test input generation
            dt = datetime(2023, 5, 1, 12, 0, 0)
            content = sim._generate_input_content(
                dt=dt,
                latitude=60.0,
                longitude=10.0,
                override_albedo=0.9
            )
            
            print(f"✓ Generated input content ({len(content)} characters)")
            
            # Check that essential parameters are present
            assert 'rte_solver twostr' in content
            assert 'albedo 0.9' in content
            assert 'wavelength 400 3600' in content
            print("✓ Input content contains expected parameters")
            
            return True
            
    except Exception as e:
        print(f"✗ Core test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_era5_support():
    """Test ERA5 atmosphere file support."""
    print("\nTesting ERA5 support...")
    
    try:
        import numpy as np
        import xarray as xr
        from pyradtran.io_unified import ERA5AtmosphereGenerator
        
        # Create mock ERA5 dataset
        pressure_levels = [1000, 850, 700, 500, 300, 200, 100]
        
        era5_ds = xr.Dataset({
            'z': (['pressure_level', 'latitude', 'longitude', 'valid_time'], 
                  np.random.random((len(pressure_levels), 1, 1, 1)) * 50000),
            't': (['pressure_level', 'latitude', 'longitude', 'valid_time'], 
                  280 + np.random.random((len(pressure_levels), 1, 1, 1)) * 40),
            'q': (['pressure_level', 'latitude', 'longitude', 'valid_time'], 
                  np.random.random((len(pressure_levels), 1, 1, 1)) * 0.02),
            'o3': (['pressure_level', 'latitude', 'longitude', 'valid_time'], 
                   np.random.random((len(pressure_levels), 1, 1, 1)) * 1e-5)
        }, coords={
            'pressure_level': pressure_levels,
            'latitude': [60.0],
            'longitude': [10.0],
            'valid_time': [pd.Timestamp('2023-05-01T12:00:00')]
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "era5_atmosphere.dat"
            
            generator = ERA5AtmosphereGenerator()
            result_path = generator.create_era5_atmosphere_file(
                era5_ds=era5_ds,
                latitude=60.0,
                longitude=10.0,
                time='2023-05-01T12:00:00',
                output_filepath=output_path
            )
            
            print("✓ ERA5 atmosphere file created")
            
            # Check file content
            content = output_path.read_text()
            lines = content.strip().split('\n')
            data_lines = [line for line in lines if not line.startswith('#')]
            
            print(f"✓ ERA5 file has {len(data_lines)} data lines")
            
            # Verify format
            for line in data_lines[:3]:  # Check first few lines
                columns = line.split()
                assert len(columns) == 9, f"Expected 9 columns, got {len(columns)}"
            
            print("✓ ERA5 file format is correct")
            
            return True
            
    except Exception as e:
        print(f"✗ ERA5 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("PYRADTRAN UNIFIED SYSTEM VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration System", test_config_system),
        ("IO System", test_io_system), 
        ("Core System", test_core_system),
        ("ERA5 Support", test_era5_support)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print(f"VALIDATION SUMMARY: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The unified system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
