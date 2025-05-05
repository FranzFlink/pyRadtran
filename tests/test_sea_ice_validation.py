# tests/test_sea_ice_validation.py
"""
Validation tests for pyradtran focusing on sea ice simulations.

These tests validate that pyradtran can properly recreate the specific
sea ice simulation scenarios from the original disort.py script.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta

from pyradtran.config import (
    PathsConfig, 
    SimulationDefaults,
    SimulationConfig,
    CloudParameters,
    AerosolParameters,
    load_config
)
from pyradtran.core import Simulation
from pyradtran.helpers import configure_surface, configure_spectral_range
from pyradtran.interface import run_pyradtran_simulation

# --- Test Configuration ---

# Use the actual paths from the user's environment
LIBRADTRAN_DATA_PATH = '/opt/libradtran/2.0.4/share/libRadtran/data'
LIBRADTRAN_EXEC_PATH = '/opt/libradtran/2.0.4/bin/uvspec'
ATMOSPHERE_FILE = '/projekt_agmwend/data/HALO-AC3/05_VELOX_Tools/add_data/afglsw.dat'
SOLAR_SPECTRUM_FILE = '/projekt_agmwend/home_rad/sophie/libradtran/solar_flux/NewGuey2003.dat'
RADIOSONDE_BASE_PATH = '/projekt_agmwend/data/HALO-AC3/01_soundings/RS_for_libradtran/Dropsondes_HALO/'
SIMULATION_OUTPUT_DIR = '/projekt_agmwend/home_rad/Joshua/HALO-AC3_Arctic_leads/data/simulation/disort/'

# Sea Ice simulation constants (from disort.py)
FIXED_OZONE_DU = 300.0  # Fixed total column ozone in Dobson Units
FIXED_IWV_MM = 2.0      # Fixed total column water vapor (precipitable water) in mm
FIXED_SURFACE_TEMP_K = 250.0  # Fixed surface temperature for sea ice conditions (Kelvin)
SEA_ICE_BRDF_TYPE = 20  # RPV BRDF type for sea ice

# Skip tests if LibRadtran is not installed
def has_libradtran():
    """Check if LibRadtran executable exists"""
    return os.path.isfile(LIBRADTRAN_EXEC_PATH) and os.path.isdir(LIBRADTRAN_DATA_PATH)

# --- Fixtures ---

@pytest.fixture
def sea_ice_config():
    """Create a config for sea ice simulations similar to disort.py"""
    return SimulationConfig(
        paths=PathsConfig(
            libradtran_bin=Path(LIBRADTRAN_EXEC_PATH),
            libradtran_data=Path(LIBRADTRAN_DATA_PATH),
            atmosphere_profile=Path(ATMOSPHERE_FILE),
            solar_spectrum=Path(SOLAR_SPECTRUM_FILE),
            radiosonde_base=Path(RADIOSONDE_BASE_PATH),
            output_dir=Path(SIMULATION_OUTPUT_DIR),
            working_dir=Path(tempfile.gettempdir())
        ),
        simulation_defaults=SimulationDefaults(
            rte_solver='twostr',  # Same as disort.py
            mol_abs_param='lowtran per_nm',  # Same as disort.py
            wavelength_nm=[400, 3600],  # Same range as disort.py
            output_columns=['sza', 'edir', 'eglo', 'edn', 'eup', 'enet', 'esum', 'albedo'],
            output_altitudes_km=[0.0],  # Surface level
            
            # Surface properties
            albedo_type='library',
            albedo_library='IGBP',
            brdf_type='rpv',
            brdf_rpv_type=SEA_ICE_BRDF_TYPE,  # Sea ice BRDF
            surface_temperature_k=FIXED_SURFACE_TEMP_K,
            
            # Fixed atmospheric composition
            mol_modify={
                'O3': {'value': FIXED_OZONE_DU, 'unit': 'DU'},
                'H2O': {'value': FIXED_IWV_MM, 'unit': 'MM'}
            },
            
            # Default aerosols
            aerosols=AerosolParameters(
                enabled=True,
                aerosol_type='default'
            ),
            
            # No clouds by default
            clouds=CloudParameters(enabled=False)
        ),
    )

@pytest.fixture
def arctic_test_dataset():
    """Create a test dataset with Arctic locations similar to disort.py usage"""
    # Create time series (one day with hourly points)
    start_time = datetime(2025, 5, 5, 0, 0, 0)
    times = [start_time + timedelta(hours=i) for i in range(24)]
    
    # Arctic locations
    lats = np.full(24, 75.0)  # Arctic
    lons = np.full(24, 0.0)   # Prime meridian
    
    # Create dataset
    ds = xr.Dataset(
        coords={
            'time': times,
            'latitude': ('time', lats),
            'longitude': ('time', lons)
        }
    )
    return ds

# --- Validation Tests ---

@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
def test_sea_ice_simulation_basic(sea_ice_config, arctic_test_dataset):
    """Test basic sea ice simulation matches expected patterns"""
    
    # Get a temporary output path
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        output_path = Path(tmp.name)
    
    try:
        # Run simulation using xarray accessor
        import pyradtran  # Import to register accessor
        
        # Run with the accessor
        result = arctic_test_dataset.pyradtran.run_uvspec(
            config=sea_ice_config,
            output_path=output_path,
            return_dataset=True
        )
        
        # Verify result is an xarray Dataset
        assert isinstance(result, xr.Dataset), "Result should be an xarray Dataset"
        
        # Verify it contains the expected variables
        for column in sea_ice_config.simulation_defaults.output_columns:
            assert column in result, f"Variable {column} missing from result"
        
        # Verify SZA follows expected diurnal pattern in Arctic
        sza_values = result.sza.values
        assert sza_values.min() < 90, "Minimum SZA should be less than 90 degrees (sun above horizon)"
        assert not np.isnan(sza_values).any(), "SZA should not contain NaN values"
        
        # Verify irradiance follows diurnal pattern 
        # (should be correlated with SZA - higher SZA means lower irradiance)
        correlation = np.corrcoef(sza_values, result.eglo.values)[0, 1]
        assert correlation < -0.9, "Strong negative correlation expected between SZA and global irradiance"
        
        # Verify night values - when SZA > 90, direct irradiance should be very low/zero
        night_indices = sza_values > 90
        if night_indices.any():  # If there are any night values
            assert np.all(result.edir.values[night_indices] < 1), "Direct irradiance at night should be near zero"
        
        # Verify output file was created
        assert output_path.exists(), f"Output file {output_path} was not created"
        
    finally:
        # Clean up
        if output_path.exists():
            os.unlink(output_path)

@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
def test_sea_ice_twostr_vs_disort_comparison(sea_ice_config, arctic_test_dataset):
    """
    Test comparing twostr and disort results for sea ice simulations.
    This validates that the solvers produce similar results for sea ice scenarios.
    """
    # Make a dataset with just one time point for efficiency
    test_ds = arctic_test_dataset.isel(time=slice(0, 1))
    
    # Run with twostr (default in sea_ice_config)
    twostr_result = test_ds.pyradtran.run_uvspec(
        config=sea_ice_config,
        return_dataset=True,
        save_to_file=False
    )
    
    # Update config to use disort
    disort_config = sea_ice_config
    disort_config.simulation_defaults.rte_solver = 'disort'
    
    # Run with disort
    disort_result = test_ds.pyradtran.run_uvspec(
        config=disort_config,
        return_dataset=True,
        save_to_file=False
    )
    
    # Verify outputs have expected variables
    for column in sea_ice_config.simulation_defaults.output_columns:
        assert column in twostr_result, f"Column {column} missing from twostr result"
        assert column in disort_result, f"Column {column} missing from disort result"
    
    # Verify SZA is the same (should be identical for same time/location)
    np.testing.assert_allclose(
        twostr_result.sza.values, 
        disort_result.sza.values, 
        rtol=1e-5, 
        err_msg="SZA should be identical between solvers"
    )
    
    # Note: For sea ice simulations, differences between solvers might be larger
    # due to complex BRDF handling, but should still be within reasonable bounds
    eglo_diff_pct = abs(disort_result.eglo.values - twostr_result.eglo.values) / disort_result.eglo.values * 100
    assert eglo_diff_pct.max() < 20, "Irradiance difference between solvers exceeds 20% for sea ice"
    
    print(f"Mean difference between disort and twostr: {eglo_diff_pct.mean():.2f}%")
    print(f"Max difference between disort and twostr: {eglo_diff_pct.max():.2f}%")

@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
def test_sea_ice_surface_properties(sea_ice_config):
    """Test that sea ice surface properties are correctly handled"""
    
    # Create dataset with a single point
    ds = xr.Dataset(
        coords={
            'time': [pd.Timestamp('2025-05-05 12:00:00')],
            'latitude': [75.0],
            'longitude': [0.0]
        }
    )
    
    # Create simulation instances with different surface properties
    
    # 1. Run with default sea ice config
    sea_ice_result = ds.pyradtran.run_uvspec(
        config=sea_ice_config,
        return_dataset=True,
        save_to_file=False
    )
    
    # 2. Create water surface config
    water_config = sea_ice_config
    water_config = configure_surface(water_config, "ocean")
    
    water_result = ds.pyradtran.run_uvspec(
        config=water_config,
        return_dataset=True,
        save_to_file=False
    )
    
    # 3. Create snow surface config
    snow_config = sea_ice_config
    snow_config = configure_surface(snow_config, "snow")
    
    snow_result = ds.pyradtran.run_uvspec(
        config=snow_config,
        return_dataset=True,
        save_to_file=False
    )
    
    # Verify surface albedo differences
    # Snow should have higher albedo than sea ice, which should be higher than water
    assert snow_result.albedo.values[0] > sea_ice_result.albedo.values[0], "Snow albedo should be higher than sea ice"
    assert sea_ice_result.albedo.values[0] > water_result.albedo.values[0], "Sea ice albedo should be higher than water"
    
    # Verify upward radiation follows albedo pattern
    assert snow_result.eup.values[0] > sea_ice_result.eup.values[0], "Snow upward radiation should be higher than sea ice"
    assert sea_ice_result.eup.values[0] > water_result.eup.values[0], "Sea ice upward radiation should be higher than water"
    
    # Print values for reference
    print(f"Snow albedo: {snow_result.albedo.values[0]:.3f}")
    print(f"Sea ice albedo: {sea_ice_result.albedo.values[0]:.3f}")
    print(f"Water albedo: {water_result.albedo.values[0]:.3f}")

@pytest.mark.skipif(not has_libradtran(), reason="LibRadtran not available")
def test_wavelength_range_effect(sea_ice_config):
    """Test the effect of different wavelength ranges on sea ice simulations"""
    
    # Create dataset with a single point
    ds = xr.Dataset(
        coords={
            'time': [pd.Timestamp('2025-05-05 12:00:00')],
            'latitude': [75.0],
            'longitude': [0.0]
        }
    )
    
    # Default range: 400-3600 nm
    default_result = ds.pyradtran.run_uvspec(
        config=sea_ice_config,
        return_dataset=True,
        save_to_file=False
    )
    
    # Visible only: 400-700 nm
    visible_config = sea_ice_config
    visible_config = configure_spectral_range(visible_config, "visible")
    
    visible_result = ds.pyradtran.run_uvspec(
        config=visible_config,
        return_dataset=True,
        save_to_file=False
    )
    
    # Near IR: 700-1400 nm
    nir_config = sea_ice_config
    nir_config = configure_spectral_range(nir_config, "nir")
    
    nir_result = ds.pyradtran.run_uvspec(
        config=nir_config,
        return_dataset=True,
        save_to_file=False
    )
    
    # Verify that total irradiance follows expected pattern
    # Default (broadband) should be larger than visible or NIR alone
    assert default_result.eglo.values[0] > visible_result.eglo.values[0], "Broadband irradiance should exceed visible-only"
    assert default_result.eglo.values[0] > nir_result.eglo.values[0], "Broadband irradiance should exceed NIR-only"
    
    # Sum of visible and NIR should be less than broadband (due to SWIR contribution)
    visible_plus_nir = visible_result.eglo.values[0] + nir_result.eglo.values[0]
    assert default_result.eglo.values[0] > visible_plus_nir, "Broadband should exceed sum of visible and NIR (due to SWIR)"
    
    # Print values for reference
    print(f"Broadband (400-3600 nm) irradiance: {default_result.eglo.values[0]:.2f} W/m²")
    print(f"Visible (400-700 nm) irradiance: {visible_result.eglo.values[0]:.2f} W/m²")
    print(f"NIR (700-1400 nm) irradiance: {nir_result.eglo.values[0]:.2f} W/m²")
    print(f"Visible + NIR: {visible_plus_nir:.2f} W/m²")