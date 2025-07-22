# Installation

## Requirements

Before installing `pyradtran`, make sure you have:

* Python 3.7 or higher
* libRadtran installed on your system

## Basic Installation

You can install `pyradtran` directly from GitHub:

```bash
git clone git@github.com:FranzFlink/pyRadtran.git
cd pyradtran
pip install -e .
```

## Installing libRadtran

`pyradtran` requires that you have libRadtran installed on your system. Follow these steps to install libRadtran:

1. Download libRadtran from http://www.libradtran.org
2. Extract the archive
3. Configure, build, and install:

```bash
gzip -d libradtran-x.yy.tar.gz
tar -xvf libradtran-x.yy.tar
./configure
make
make check
sudo make install
```

## Environment Configuration

Make sure that the libRadtran executables are in your PATH. You can add the following to your `.bashrc` or `.bash_profile`:

```bash
export PATH=$PATH:/path/to/libradtran/bin
```
