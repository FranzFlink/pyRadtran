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
| `sza` | > 90° means the sun is below the horizon → all-zero solar output |
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
