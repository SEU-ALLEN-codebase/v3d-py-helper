"""
.. include:: ../README.md
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("v3d-py-helper")
except PackageNotFoundError:
    __version__ = "UNKNOWN"
