# Configuration

pyRadtran assembles its configuration from **three YAML layers** — later
layers override earlier ones:

1. **Package defaults** — sensible values shipped with pyRadtran
2. **Your master config** — `~/.pyradtran/config.yaml`: machine-specific
   paths, set once (see {doc}`/installation`)
3. **Simulation config** — the YAML you pass as `config_path=`: only what
   is *different* for this experiment

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

You can also override any config field per run without touching YAML,
using a dotted key in `params`:

```python
ds.pyradtran.run(
    config_path='config/my_simulation.yaml',
    params={'simulation_defaults.wavelength_nm': [400, 700]},
)
```

## Which knob lives where?

- **Config**: everything that defines the *experiment* — solver, spectral
  range, output columns/altitudes, parallelism, paths
- **`params`**: everything that varies *at run time* — per-point values,
  parameter sweeps, extra uvspec options (see {doc}`parameters`)

Both end up in the same generated input file;
`ds.pyradtran.explain()` shows exactly which layer produced each line.
