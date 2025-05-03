from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="flow_cy",
        sources=["flow_cy.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "language_level": 3
        }
    )
)
