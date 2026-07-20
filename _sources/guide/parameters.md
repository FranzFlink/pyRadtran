# Parameters & Validation

All runtime parameters go through one mapping: `params`. Any libRadtran
option can be a key; the value decides how it is applied.

## The `params` mapping

A key can be:

1. **A uvspec option** (`albedo`, `sza`, `zout`, `wc_modify`,
   `ic_properties`, … — anything your libRadtran accepts)
2. **A dotted config path** (`simulation_defaults.wavelength_nm`) —
   applied to the configuration instead of the input file

The value can be:

- **A literal** — same value for every simulated point
- **`Var("name")`** — resolved per point from the variable `name` in
  your input dataset

```python
from pyradtran import Var

ds_sim = ds.pyradtran.run(
    config_path='config/solar.yaml',
    params={
        'albedo': Var('surface_albedo'),   # per-point from the dataset
        'mol_modify O3': 320.0,            # same for every point
        'wc_modify': ['tau550 set 12'],    # repeatable option: list
        'aerosol_default': True,           # flag option: bare keyword
    },
)
```

Points where a `Var` value is NaN simply omit that parameter (the config
default applies); NaN *coordinates* or a missing/NaT *time* skip the
whole point and record `status=2` in the result.

Flag options take `True` (emit the bare keyword) or `False` (emit
nothing — useful to switch a config-supplied flag off for one run).
Numeric `zout` values are sorted and de-duplicated automatically:
uvspec hard-errors on unsorted output altitudes.

## Validated against *your* libRadtran

libRadtran ships a machine-readable description of every option. On
first use pyRadtran extracts it from your local installation (cached
under `~/.pyradtran/`), so every entry is checked against the exact
binary you run — *before* any simulation starts:

```python
params={'albdeo': 0.3}
# ValidationError: 'albdeo' is not a known uvspec option of the local
# libRadtran install; did you mean albedo / albedo_map?

params={'ic_properties': 'granite'}
# ValidationError: 'ic_properties': 'granite' is not one of
# ['baum', 'baum_v36', 'echam4', 'fu', 'hey', 'key', ...]

params={'cloudcover wc': 1.5}
# ValidationError: 'cloudcover wc' value 1.5 outside [0.0, 1.0]
```

Need an option your schema does not know (patched uvspec build)? Wrap it
in `Raw` to bypass validation:

```python
from pyradtran import Raw
params={'my_custom_option': Raw('anything goes')}
```

## Explore all options from Python

```python
import pyradtran

print(pyradtran.describe('wc_modify'))
# wc_modify <gg|ssa|tau|tau550> <set|scale> <float>
# group: Water and ice clouds   (repeatable)
# requires: wc_file
# ... full documentation ...

pyradtran.search_options('optical thickness')
# ['aerosol_modify', 'ic_modify', 'wc_modify', ...]
```

`describe()` shows the usage signature, valid choices and ranges, option
dependencies, and the full documentation text — for the libRadtran
version you actually have installed.

## Preview before you run

`explain()` renders the exact input file for one point — without running
anything — and tags every line with the layer that produced it:

```python
print(ds.pyradtran.explain(
    params={'albedo': Var('surface_albedo')},
    config_path='config/solar.yaml',
))
# rte_solver disort          # config
# albedo 0.85                # dataset-var
# mol_modify O3 320.0 DU     # params-literal
# ...
```

The {doc}`/notebooks/parameters_deep_dive` notebook walks through all of
this hands-on.

```{note}
The old `parameter_overrides=`, `albedo_var=`, `surface_temperature_var=`,
`surface_type_var=` and `altitude_var=` keyword arguments still work but
are deprecated: `albedo_var="x"` → `params={"albedo": Var("x")}`,
`parameter_overrides={...}` → `params={...}`.
```
