from io import open
from os import path

from setuptools import find_namespace_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pybio_torch",
    version="0.1a",
    description="Common torch based components for bioimage zoo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bioimage-io/pytorch-bioimage-io",
    author="Bioimage Team",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_namespace_packages(exclude=["tests"]),  # Required
    install_requires=[
        "torch>=1.1",
        "numpy>=1.17",
        "imageio>=2.5",
        "pybio@git+http://github.com/bioimage-io/python-bioimage-io",
    ],
    extras_require={"test": ["tox", "pytest"]},
    project_urls={  # Optional
        "Bug Reports": "https://github.com/bioimage-io/pytorch-bioimage-io/issues",
        "Source": "https://github.com/bioimage-io/pytorch-bioimage-io",
    },
)
