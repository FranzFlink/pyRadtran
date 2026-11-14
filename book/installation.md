# Installation

## Requirements

* Python 3.9 or higher
* [libRadtran](http://www.libradtran.org)
* numpy, pandas and xarray (installed with pyRadtran)

## Install from source

```bash
git clone https://github.com/FranzFlink/pyRadtran.git
# or: git clone git@github.com:FranzFlink/pyRadtran.git

cd pyRadtran
pip install -e .
```

## Install libRadtran

1. Download libRadtran from <http://www.libradtran.org>
2. Extract the archive and build:

```bash
tar -xzvf libradtran-x.yy.tar.gz
cd libradtran-x.yy
./configure
make
make check
```

3. Note the installation path for the next step.

## Master configuration

The master config `~/.pyradtran/config.yaml` tells pyRadtran where
libRadtran is. Every simulation config inherits from it
({doc}`guide/configuration`). Create it from Python:

```python
import pyradtran

pyradtran.save_master_config(
    libradtran_bin="/opt/libradtran/bin/uvspec",
    libradtran_data="/opt/libradtran/share/libRadtran/data",
)
# Master config saved to: ~/.pyradtran/config.yaml
```

or write the file by hand:

```yaml
# ~/.pyradtran/config.yaml
paths:
  libradtran_bin: /opt/libradtran/bin/uvspec
  libradtran_data: /opt/libradtran/share/libRadtran/data
  atmosphere_profile: afglms        # short name or full path
  solar_spectrum: NewGuey2003       # short name or full path
```

`atmosphere_profile` and `solar_spectrum` accept short names (`afglus`,
`afglms`, …, `kurudz_1.0nm`, `NewGuey2003`, …) that resolve inside
`libradtran_data`; `pyradtran.list_atmosphere_profiles()` and
`pyradtran.list_solar_spectra()` list them.

## Optional dependencies

| Package | Purpose |
|---------|---------|
| `siphon` | radiosonde retrieval from IGRA |
| `gcsfs` | ERA5 access from Google Cloud |
| `scikit-learn` | statistical metrics in the validation notebooks |
| `seaborn` | plotting in some notebooks |

```bash
pip install siphon gcsfs scikit-learn seaborn
```

## Verify

```python
import pyradtran
print(pyradtran.__version__)
```

Then run the {doc}`quickstart notebook <notebooks/quickstart>` to
confirm that pyRadtran finds libRadtran.
