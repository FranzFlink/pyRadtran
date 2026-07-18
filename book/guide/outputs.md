# Results & Post-Processing

## The result dataset

Results come back as an `xarray.Dataset` with your input coordinates plus
`wavelength` / `altitude` dimensions where applicable:

```python
ds_sim = ds.pyradtran.run(config_path=cfg)

ds_sim.edir                                  # direct irradiance
ds_sim.eglo.sel(wavelength=550, method='nearest').plot()
```

Which variables appear is set by `output_columns` in your config (the
uvspec `output_user` line). You never need to list `lambda` or `zout`
yourself — pyRadtran adds them automatically for spectral and
multi-altitude runs so the wavelength/altitude axes can always be
reconstructed.

With `save_to_file=True` (default) the dataset is also written to a
CF-style NetCDF file with the full configuration embedded as metadata.

## Status codes & failure logs

Every result carries a per-point `status` variable:

| value | meaning |
|-------|---------|
| 0 | ok |
| 1 | uvspec failed |
| 2 | skipped (NaN coordinates) |

When at least one point fails, a `failures_<timestamp>.log` with the
captured stderr is written to the working directory, and the failing
`.inp` file is kept for post-mortem (see {doc}`debugging`).

## Instrument channels

Convolve spectral results with instrument spectral response functions
(SRFs) — either as post-processing or directly in `run()`:

```python
from pyradtran import convolve_channels, brightness_temperature

# srf: DataArray with dims (channel, wavelength), wavelength in nm
channel_ds = ds.pyradtran.run(config_path=cfg, channels=srf)

# thermal radiances -> brightness temperature (uvspec default units)
tb = brightness_temperature(channel_ds['uu'], wavelength_nm=10500.0)
```

`keep_spectral=True` retains the original spectral variables as
`<name>_spectral` next to the channel-averaged ones.

## Sensitivity kernels

`jacobian()` runs the batch twice (base and perturbed) and returns the
finite-difference kernel `(perturbed - base) / delta`:

```python
jac = ds.pyradtran.jacobian('albedo', 0.01, config_path=cfg)
jac['eup']    # d(eup)/d(albedo), same dims as a normal result
```

The perturbed parameter must be a scalar (a `params` literal or config
default) — perturbing a per-point `Var` is rejected.
