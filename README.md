# Vaa3D in Python Made Easy
Python library for Vaa3D functions.

## Installation

```shell
$ pip install v3d-py-helper
```

By cloning the repo and test the Cyhton usages:
```shell
$ python setup.py build_ext --inplace
```

## Usage

### Loading Vaa3D format data

```python
from v3dpy.loaders import Raw, PBD

raw = Raw()
img = raw.load('path.v3draw')
raw.save('path.v3draw', img)

pbd = PBD()
img = pbd.load('path.v3dpbd')
pbd.save('path.v3dpbd', img)
```

## Useful Links

Github project: https://github.com/SEU-ALLEN-codebase/v3d-py-helper

Vaa3D source: https://github.com/Vaa3D/v3d_external

Documentation: https://SEU-ALLEN-codebase.github.io/v3d-py-helper