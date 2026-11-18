# Configuration

The configuration is merged from **three YAML layers**; later layers
override earlier ones:

1. **Package defaults**, shipped with pyRadtran
2. **Master config** `~/.pyradtran/config.yaml`: machine-specific
   paths, set once (see {doc}`/installation`)
3. **Simulation config**, the YAML passed as `config_path=`: only what
   differs for this experiment

A typical simulation config is short:

```yaml
simulation_defaults:
  source: solar
  rte_solver: disort
  mol_abs_param: lowtran per_nm
  wavelength_nm: [400, 770]     # nm; {start: 400, end: 770} also accepted
  output_columns: [eglo, eup, edir]
  output_altitudes_km: [0.0]

execution:
  max_workers: 8              # parallel uvspec processes
  timeout_seconds: 60
  cleanup_temp_files: false   # keep .inp files for debugging
```

Paths (`libradtran_bin`, `libradtran_data`, `atmosphere_profile`,
`solar_spectrum`) belong in the master config and are inherited by every
simulation.

## Building configs in Python

```python
import pyradtran

cfg = pyradtran.load_config()               # defaults + master config
cfg.simulation_defaults.albedo_value = 0.2
cfg.to_yaml('config/my_simulation.yaml')    # save for reuse
```

A dotted key in `params` overrides a config field for one run:

```python
ds.pyradtran.run(
    config_path='config/my_simulation.yaml',
    params={'simulation_defaults.wavelength_nm': [400, 700]},
)
```

## Config or `params`?

- **Config**: what defines the experiment (solver, spectral range,
  output columns and altitudes, parallelism, paths)
- **`params`**: what varies at run time (per-point values, parameter
  sweeps, extra uvspec options; see {doc}`parameters`)

Both end up in the same input file; `ds.pyradtran.explain()` shows
which layer produced each line.

## Field reference

All fields of a simulation config, by section. Details:
{class}`~pyradtran.config.SimulationDefaults`,
{class}`~pyradtran.config.PathsConfig`,
{class}`~pyradtran.config.ExecutionConfig`,
{class}`~pyradtran.config.OutputConfig`.

### `simulation_defaults`

| Field | uvspec line it produces | Notes |
|---|---|---|
| `source` | `source solar <spectrum>` / `source thermal` | `solar` or `thermal` |
| `rte_solver` | `rte_solver …` | `twostr` (fast, irradiances), `disort` (accurate, radiances, clouds) |
| `mol_abs_param` | `mol_abs_param …` | `lowtran`, `reptran coarse\|medium\|fine`, `kato2`, `fu`, … |
| `wavelength_nm` | `wavelength <min> <max>` | nm; `[min, max]` |
| `integrate_wavelength` | `output_process integrate` | `true` → one broadband value, no `wavelength` dimension |
| `output_columns` | `output_user …` | see {doc}`outputs`; `lambda` / `zout` are added automatically |
| `output_altitudes_km` | `zout …` | km; sorted and de-duplicated; overridden by an `altitude` coordinate in the dataset |
| `albedo_value` | `albedo …` | 0–1; overridden by `params["albedo"]` |
| `surface_temperature_k` | `sur_temperature …` | K; thermal source only |
| `ozone_du` | `mol_modify O3 … DU` | default `null` (profile column); skipped when the atmosphere profile brings its own ozone |
| `h2o_mm` | `mol_modify H2O … MM` | precipitable water; default `null` (profile column); skipped when a radiosonde/ERA5 profile provides humidity, also one passed through `params` |
| `h2o_source` | — | `fixed` or `radiosonde` |
| `sza` | `sza …` | degrees; computed from time and location when unset |
| `viewing_geometry` | `umu 1.0` | `nadir` |
| `clouds` | `wc_file` / `ic_file` | one static cloud layer (or cloud files) for every point; for anything per-point use the run-time mechanisms in {doc}`clouds` |

### `paths`

| Field | Meaning |
|---|---|
| `libradtran_bin` | the `uvspec` executable |
| `libradtran_data` | libRadtran's `data/` directory (`data_files_path`) |
| `atmosphere_profile` | background atmosphere: short name (`afglus`, `afglss`, …) or path |
| `solar_spectrum` | extraterrestrial spectrum: short name (`NewGuey2003`, `kurudz_1.0nm`, …) or path |
| `working_dir` | where `.inp` / `.out` files and failure logs go (default `./pyradtran_work`) |
| `output_dir` | where NetCDF results go (default `./pyradtran_output`) |
| `radiosonde_base` | root directory of local radiosonde files |

### `execution`

| Field | Meaning |
|---|---|
| `max_workers` | parallel uvspec processes |
| `timeout_seconds` | per-point time limit (default 300) |
| `cleanup_temp_files` | delete `.inp`/`.out` of successful points (failed points are always kept) |
| `debug_mode` | verbose logging |

### `output`

| Field | Meaning |
|---|---|
| `filename_prefix`, `filename_suffix` | auto-generated name: `<prefix>_<timestamp><suffix>` |
| `netcdf_encoding` | compression settings passed to xarray |

## Choosing solver and spectral resolution

| Goal | `rte_solver` | `mol_abs_param` |
|---|---|---|
| quick look, clear-sky irradiances | `twostr` | `lowtran` |
| broadband fluxes, clouds | `disort` | `reptran coarse` (or `kato2` solar / `fu` thermal) |
| narrow-band instrument channels | `disort` | `reptran medium` / `fine` |
| radiances (`uu`), brightness temperatures | `disort` | `reptran fine` |

```{warning}
`twostr` is a quick-look solver. With libRadtran 2.0.6 its upward
irradiance at a reflecting surface is
$E^\uparrow = \tfrac{2}{\sqrt{3}}\,\alpha\,E_\mathrm{glo}$, about
15 % above the Lambertian value $\alpha\,E_\mathrm{glo}$ that `disort`
and `rodents` return. This is a property of uvspec, not of pyRadtran. Use `disort` (or `rodents` for a fast two-stream
alternative) whenever `eup`, `enet` or the albedo matter.
```

`pyradtran.describe("mol_abs_param")` and
`pyradtran.describe("rte_solver")` print the libRadtran
documentation of the installed version.
