# Configuration Gallery

This gallery showcases various simulation configurations available in pyRadtran.

## Available Configurations

### Surface Properties

#### Albedo Configuration
**File:** `albedo.yaml`

```{literalinclude} albedo.yaml
:language: yaml
:caption: Albedo simulation configuration
```

---

### Cloud Simulations

#### Solar Cloud Simulation
**File:** `cloud_solar.yaml`

```{literalinclude} cloud_solar.yaml
:language: yaml
:caption: Solar cloud simulation configuration
```

---

#### Thermal Cloud Simulation
**File:** `cloud_thermal.yaml`

```{literalinclude} cloud_thermal.yaml
:language: yaml
:caption: Thermal cloud simulation configuration
```

---

#### Realistic Cloud Example
**File:** `realistic_cloud_example.yaml`

```{literalinclude} realistic_cloud_example.yaml
:language: yaml
:caption: Realistic cloud simulation configuration
```

---

### Atmospheric Profiles

#### Radiosonde Configuration
**File:** `radiosonde.yaml`

```{literalinclude} radiosonde.yaml
:language: yaml
:caption: Radiosonde atmospheric profile configuration
```

---

#### Radiosonde Thermal Configuration
**File:** `radiosonde_thermal.yaml`

```{literalinclude} radiosonde_thermal.yaml
:language: yaml
:caption: Radiosonde thermal simulation configuration
```

---

### Spectral and Thermal Simulations

#### Spectral Configuration
**File:** `spectral_config.yaml`

```{literalinclude} spectral_config.yaml
:language: yaml
:caption: Spectral simulation configuration
```

---

#### Thermal Configuration
**File:** `thermal_config.yaml`

```{literalinclude} thermal_config.yaml
:language: yaml
:caption: Thermal simulation configuration
```

---

### Field Campaign

#### HALO-AC3 BBR Configuration
**File:** `halo-ac3_bbr_all_aircraft.ipynb.yaml`

```{literalinclude} halo-ac3_bbr_all_aircraft.ipynb.yaml
:language: yaml
:caption: HALO-AC3 BBR aircraft simulation configuration
```

---

#### Velox Configuration
**File:** `velox.yaml`

```{literalinclude} velox.yaml
:language: yaml
:caption: Velox simulation configuration
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
