# Debugging

(check-your-input-file)=
## Check your input file!

The fastest way to diagnose a radiative transfer problem is to look at
the exact input `uvspec` received. Two ways:

**Before running** — `explain()` renders the annotated input file for one
point without executing anything:

```python
print(ds.pyradtran.explain(params={...}, config_path=cfg))
# rte_solver disort          # config
# albedo 0.85                # dataset-var
# ...
```

**After running** — keep the generated `.inp` files:

```yaml
execution:
  cleanup_temp_files: false
```

then look in the working directory (default `pyradtran_work/`), one
`.inp` (and one raw `.out`) per simulation point. Failed points keep
their input file — and any generated cloud profile files it references —
even with cleanup enabled, and the captured stderr lands in
`failures_<timestamp>.log`. With `cleanup_temp_files: true`, successful
runs remove both their `.inp` and `.out` files once parsed.

## What to check

| Line | What to look for |
|------|-----------------|
| `atmosphere_file`, `source solar` | Do the paths exist? Right atmosphere model? |
| `wavelength` | Range in nm, supported by your `mol_abs_param`? |
| `sza` | > 90° means the sun is below the horizon: `disort` returns zeros, `twostr` fails (`status == 1`) |
| `zout` | Output altitudes in km, above the site's `altitude`? |
| `wc_file` / `ic_file` | Cloud file exists? Columns are z(km), LWC/IWC(g/m³), r_eff(µm)? |

Typos in option names or invalid values are caught before any run by the
schema validation (see {doc}`parameters`) — if something still fails, it
is usually a path or a physics mismatch.

## Verbose logging

```python
import logging
logging.getLogger('pyradtran').setLevel(logging.DEBUG)
```

or `execution.debug_mode: true` in the config: shows generated files,
executed commands, and raw uvspec output.

## Common errors and what they mean

Messages from the failure log (`failures_<timestamp>.log`) or raised by
pyRadtran before anything runs:

| Message | Cause | Fix |
|---|---|---|
| `ValidationError: 'albdeo' is not a known uvspec option … did you mean albedo` | typo in a `params` key | fix the key; `pyradtran.search_options()` finds the right name |
| `ValidationError: … value 1.5 outside [0.0, 1.0]` | value outside the range libRadtran accepts | check `pyradtran.describe(option)` |
| `ValidationError: … is not a configuration field; did you mean …` | typo in a dotted config key | fix the key ({doc}`configuration` lists all fields) |
| `ValidationError: … matches a dataset variable name; use Var(...)` | a bare string that is also a variable name | wrap it: `Var("name")` |
| `required start wavelength: … is not in extraterrestrial solar flux file` | `wavelength_nm` reaches beyond the solar spectrum file | narrow the range or pick another `solar_spectrum` (`pyradtran.list_solar_spectra()`) |
| `twostr_check_inputs--input and/or dimension errors` | usually `sza > 90°` with `twostr` (night-time points) | use `rte_solver: disort`, or drop night points |
| `NaN` in broadband results with clouds | `twostr` instability in optically thick layers | `rte_solver: disort` |
| all values `NaN`, `status == 0` | output columns and parser disagree — typically a hand-written `output_user` | set `output_columns` in the config instead and let pyRadtran add `lambda`/`zout` |
| `PyRadtranError: All simulations failed` | every point failed — almost always one systematic problem | read the first entry of the failure log, run that `.inp` by hand |
| `ConfigurationError`: LibRadtran executable / data directory not found | master config points to the wrong place | re-run `pyradtran.save_master_config(...)` ({doc}`/installation`) |
| run takes forever | fine spectral resolution over a wide range, or too many sweep points | coarser `mol_abs_param` (`reptran coarse`), `integrate_wavelength`, raise `max_workers` |

## Run a failing input by hand

Nothing beats the original error message. The kept `.inp` file is a
complete uvspec input:

```bash
cd pyradtran_work
/opt/libradtran/bin/uvspec < tmpab12cd34.inp
```

Remove the `quiet` line (or add `verbose`) to make uvspec explain every
step it takes — which profile it read, how it interpolated, what the
optical depths are.

## Sanity checks worth a minute

- **Energy conservation**: at the surface `eup / (edir + edn)` must
  reproduce the albedo you set. Check this with `rte_solver: disort` —
  the two-stream solver's surface `eup` does not satisfy it exactly.
- **Top of atmosphere**: the downward irradiance at 100 km should be
  $\mu_0 E_0$ — about 1361 W m⁻² × $\cos\theta_0$ for a full solar
  spectrum, scaled by the Sun–Earth distance.
- **Thermal, opaque limit**: in a strong absorption band the brightness
  temperature must approach the air temperature at the sensor.
- **Units**: broadband solar results in mW m⁻², thermal in W m⁻²
  ({ref}`units`).
