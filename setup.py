from setuptools import Extension, setup
import numpy as np


setup(
    ext_modules=[
        Extension(
            name="v3dpy.loaders.pbd",
            sources=["v3dpy/loaders/pbd.pyx"],
            include_dirs=[np.get_include()]
        ),
        Extension(
            name="v3dpy.loaders.raw",
            sources=["v3dpy/loaders/raw.pyx"],
            include_dirs=[np.get_include()]
        )
    ]
)