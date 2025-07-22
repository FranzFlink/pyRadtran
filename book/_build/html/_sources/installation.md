# Installation

## Requirements

PyRadtran requires:
- Python 3.8 or later
- libRadtran radiative transfer model
- NumPy, xarray, PyYAML

## Basic Installation

Install PyRadtran using pip:

```bash
pip install pyradtran
```

Or for development:

```bash
git clone https://github.com/FranzFlink/pyRadtran.git
cd pyRadtran
pip install -e .
```

## Installing libRadtran

PyRadtran requires the libRadtran radiative transfer model to be installed separately. 

Download and install libRadtran from: http://www.libradtran.org/

## Environment Configuration

Set the `LIBRADTRAN_PATH` environment variable:

```bash
export LIBRADTRAN_PATH=/path/to/libradtran
```

## Verification

Test your installation:

```python
import pyradtran
print(pyradtran.__version__)
```
