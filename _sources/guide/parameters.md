# Parameters & Validation

All runtime parameters go through one mapping: `params`. Any libRadtran
option can be a key; the value decides how it is applied.

## The `params` mapping

A key can be:

1. **A uvspec option** (`albedo`, `sza`, `zout`, `wc_modify`,
   `ic_properties`, …): anything your libRadtran accepts
2. **A dotted config path** (`simulation_defaults.wavelength_nm`),
   applied to the configuration instead of the input file

The value can be:

- **A literal**: the same value for every point
- **`Var("name")`**: resolved per point from the variable `name` in
  the input dataset

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

A point where a `Var` value is NaN omits that parameter (the config
default applies). NaN *coordinates* or a missing/NaT *time* skip the
whole point and record `status=2` in the result.

Flag options take `True` (emit the bare keyword) or `False` (emit
nothing, which switches a config-supplied flag off for one run).
Numeric `zout` values are sorted and de-duplicated, because uvspec
rejects unsorted output altitudes.

## Value forms

| You write | Input file gets | Tag in `explain()` |
|---|---|---|
| `"albedo": 0.3` | `albedo 0.3` for every point | `params-literal` |
| `"albedo": Var("a")` | `albedo <ds["a"] at that point>` | `dataset-var` |
| `"mol_modify O3": 320.0` | `mol_modify O3 320.0 DU` (multi-word key) | `params-literal` |
| `"aerosol_default": True` | `aerosol_default` (bare flag) | `params-literal` |
| `"aerosol_default": False` | nothing; removes a config-supplied line | — |
| `"wc_modify": ["tau550 set 12", "ssa scale 0.9"]` | one line per entry (repeatable option) | `params-literal` |
| `"zout": [0, 2, 10]` | `zout 0 2 10` (space-joined, sorted) | `params-literal` |
| `"wc_file": {"z": …, "lwc": …, "reff": …}` | `wc_file 1D <generated file>` at run time (`explain()` shows the dict) | `params-literal` |
| `"my_option": Raw("x y")` | `my_option x y`, no validation | `unvalidated` |
| `"simulation_defaults.sza": 30.0` | changes the config, not a line of its own | `config` |

Some common keys have curated units and ranges on top of the libRadtran
schema: `albedo`, `sza`, `sur_temperature`, `zout`, `mol_modify O3`
(DU), `mol_modify H2O` (mm), `umu`, `day_of_year`, … The full table is
in the {mod}`pyradtran.params` reference.

```{note}
Switch between solar and thermal with the config field
(`"simulation_defaults.source": "thermal"`), not with the uvspec option
`"source"`: only the config field also adjusts the dependent lines
(solar spectrum, `sza`, `sur_temperature`).
```

## Validation against the installed libRadtran

libRadtran ships a machine-readable description of every option. On
first use pyRadtran extracts it from the local installation (cached
under `~/.pyradtran/`) and checks every `params` entry against it
before any simulation starts:

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

For an option the schema does not know (a patched uvspec build), wrap
the value in `Raw` to bypass validation:

```python
from pyradtran import Raw
params={'my_custom_option': Raw('anything goes')}
```

## Look up options

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

`describe()` shows the usage signature, valid choices and ranges,
dependencies and the documentation text of the installed libRadtran
version.

## Preview the input file

`explain()` renders the input file for one point without running it and
tags every line with the layer that produced it:

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

The {doc}`/notebooks/parameters_deep_dive` notebook works through all
of this.

```{note}
The old `parameter_overrides=`, `albedo_var=`, `surface_temperature_var=`,
`surface_type_var=` and `altitude_var=` keyword arguments still work but
are deprecated: `albedo_var="x"` → `params={"albedo": Var("x")}`,
`parameter_overrides={...}` → `params={...}`.
```
