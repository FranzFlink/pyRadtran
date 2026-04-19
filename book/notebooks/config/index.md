# Configuration Gallery

This gallery showcases the simulation configurations used in the pyRadtran notebooks.

## Available Configurations

### Quickstart

#### Quickstart Configuration
**File:** `quickstart.yaml`

```{literalinclude} quickstart.yaml
:language: yaml
:caption: Quickstart simulation configuration
```

---

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

#### Arctic Cloud Experiment — Solar
**File:** `arctic_cloud_experiment_solar.yaml`

```{literalinclude} arctic_cloud_experiment_solar.yaml
:language: yaml
:caption: Arctic cloud experiment solar configuration
```

---

#### Arctic Cloud Experiment — Thermal
**File:** `arctic_cloud_experiment_thermal.yaml`

```{literalinclude} arctic_cloud_experiment_thermal.yaml
:language: yaml
:caption: Arctic cloud experiment thermal configuration
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

#### Radiosonde Solar Spectral Configuration
**File:** `radiosonde_solar_spectral.yaml`

```{literalinclude} radiosonde_solar_spectral.yaml
:language: yaml
:caption: Radiosonde solar spectral simulation configuration
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

### Field Campaigns

#### HALO-AC3 BBR Aircraft Configuration
**File:** `halo-ac3_bbr_all_aircraft.yaml`

```{literalinclude} halo-ac3_bbr_all_aircraft.yaml
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
