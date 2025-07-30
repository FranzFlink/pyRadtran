# Configuration Gallery

This gallery showcases various simulation configurations available in pyRadtran. Each configuration file demonstrates different use cases and parameter settings for radiative transfer simulations.

## Available Configurations

### Standard Simulations

#### Default Simulation
**File:** `default_simulation.yaml`

A comprehensive default configuration demonstrating standard solar radiative transfer simulations with typical atmospheric and surface parameters.

```{literalinclude} default_simulation.yaml
:language: yaml
:caption: Default simulation configuration
```

---

#### Thermal Simulation
**File:** `thermal_config.yaml`

Configuration optimized for thermal infrared radiative transfer simulations.

```{literalinclude} thermal_config.yaml
:language: yaml
:caption: Thermal simulation configuration
```

---

#### Thermal Simulation Example
**File:** `thermal_simulation_example.yaml`

Detailed example configuration for thermal radiative transfer calculations.

```{literalinclude} thermal_simulation_example.yaml
:language: yaml
:caption: Thermal simulation example
```

---

### Surface Property Studies

#### Albedo Configuration
**File:** `albedo.yaml`

Configuration focused on surface albedo studies and spectral reflectance analysis.

```{literalinclude} albedo.yaml
:language: yaml
:caption: Albedo simulation configuration
```

---

### Atmospheric Profile Studies

#### Radiosonde Configuration
**File:** `radiosonde.yaml`

Configuration for using radiosonde atmospheric profiles in simulations.

```{literalinclude} radiosonde.yaml
:language: yaml
:caption: Radiosonde-based atmospheric profile configuration
```


### Spectral Configurations

#### Standard Spectral Range
**File:** `spectral_config.yaml`

Configuration for standard spectral range radiative transfer simulations.

```{literalinclude} spectral_config.yaml
:language: yaml
:caption: Standard spectral configuration
```

---

#### Extended UV-NIR Spectral Range
**File:** `spectral_config_200_3600.yaml`

Configuration covering extended spectral range from 200 to 3600 nm.

```{literalinclude} spectral_config_200_3600.yaml
:language: yaml
:caption: Extended spectral range (200-3600 nm)
```

---

#### Visible-NIR Spectral Range
**File:** `spectral_config_400_3200.yaml`

Configuration for visible to near-infrared spectral range (400-3200 nm).

```{literalinclude} spectral_config_400_3200.yaml
:language: yaml
:caption: Visible-NIR spectral range (400-3200 nm)
```

---

### Cloud Simulations

#### ERA5-based Cloud Simulation
**File:** `cloud_era5_example.yaml`

Configuration for generating cloud files automatically from ERA5 datasets, demonstrating seamless integration with the xarray/ERA5 ecosystem.

```{literalinclude} cloud_era5_example.yaml
:language: yaml
:caption: ERA5-based cloud simulation configuration
```

---

#### Parametric Cloud Simulation
**File:** `cloud_parametric_example.yaml`

Configuration for defining clouds using simple layer parameters, allowing precise control over cloud properties.

```{literalinclude} cloud_parametric_example.yaml
:language: yaml
:caption: Parametric cloud simulation configuration
```

---

#### File-based Cloud Simulation
**File:** `cloud_file_example.yaml`

Configuration for using existing libRadtran cloud files, including support for mixed water/ice cloud scenarios.

```{literalinclude} cloud_file_example.yaml
:language: yaml
:caption: File-based cloud simulation configuration
```

---



## How to Use These Configurations

1. **Copy a configuration file** that matches your simulation requirements
2. **Modify the paths** to match your system setup
3. **Adjust parameters** according to your specific study
4. **Run the simulation** using pyRadtran interface

## Configuration Structure

All configuration files follow a common structure with the following main sections:

- **`paths`**: File paths for libradtran installation, data files, and output directories
- **`simulation_defaults`**: Default parameters for radiative transfer calculations
- **`execution`**: Runtime and performance settings
- **`output`**: Output format and naming conventions

For detailed information about each parameter, refer to the [Usage](../usage.md) documentation.
