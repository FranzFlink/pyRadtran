# Installation

## Requirements

Before installing pyRadtran, make sure you have:

* Python 3.9 or higher
* [libRadtran](http://www.libradtran.org) installed on your system
* Common scientific Python packages (numpy, pandas, xarray) — installed automatically with pyRadtran

## Install from Source

```bash
# Using SSH (if you have SSH keys set up)
git clone git@github.com:FranzFlink/pyRadtran.git
# or standard HTTPS
# git clone https://github.com/FranzFlink/pyRadtran.git

cd pyRadtran
pip install -e .
```

## Installing libRadtran

pyRadtran requires a working libRadtran installation. Follow these steps:

1. Download libRadtran from <http://www.libradtran.org>
2. Extract the archive and build:

```bash
tar -xzvf libradtran-x.yy.tar.gz
cd libradtran-x.yy
./configure
make
make check
```

3. Note the installation path — you'll need it to configure pyRadtran.

## Setting Up Your Master Configuration

pyRadtran uses a **layered configuration system**. The most important first step after installation is creating your personal master configuration file. This tells pyRadtran where to find libRadtran on your machine:

```bash
mkdir -p ~/.pyradtran
```

Create `~/.pyradtran/config.yaml` with the following content, adjusting paths to match your libRadtran installation:

```yaml
# ~/.pyradtran/config.yaml — User-specific configuration
# This file is loaded automatically and overrides built-in defaults.
# Adjust the paths below to match your local libRadtran installation.

paths:
  libradtran_bin: /opt/libradtran/bin/uvspec
  libradtran_data: /opt/libradtran/share/libRadtran/data
  atmosphere_profile: /opt/libradtran/share/libRadtran/data/atmmod/afglms.dat
  solar_spectrum: /opt/libradtran/share/libRadtran/data/solar_flux/NewGuey2003.dat
```

```{tip}
Once your master config is set, you typically don't need to specify libRadtran paths in individual
simulation configs — they inherit from your master config automatically.
```

## Optional Dependencies

Some pyRadtran features require additional packages:

| Package | Purpose | Install |
|---------|---------|---------|
| `siphon` | Radiosonde data retrieval from IGRA | `pip install siphon` |
| `gcsfs` | ERA5 data access from Google Cloud | `pip install gcsfs` |
| `scikit-learn` | Statistical metrics for validation notebooks | `pip install scikit-learn` |
| `seaborn` | Enhanced plotting in some notebooks | `pip install seaborn` |

```bash
# Install all optional dependencies at once
pip install siphon gcsfs scikit-learn seaborn
```

## Verifying Your Installation

After installation, verify everything works:

```python
import pyradtran
print(pyradtran.__version__)
```

Then try running the albedo quickstart notebook to confirm that pyRadtran can communicate with libRadtran.
