from pathlib import Path

from setuptools import find_namespace_packages, setup

# Get the long description from the README file
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="pybio.torch",
    version="0.3a",
    description="Common torch based components for bioimage zoo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bioimage-io/pytorch-bioimage-io",
    author="Bioimage Team",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_namespace_packages(exclude=["tests"]),  # Required
    install_requires=[
        "torch>=1.4",
        "numpy>=1.17",
        "pybio.core @ git+http://github.com/bioimage-io/python-bioimage-io#egg=pybio.core",
    ],
    extras_require={"test": ["tox", "pytest"]},
    project_urls={  # Optional
        "Bug Reports": "https://github.com/bioimage-io/pytorch-bioimage-io/issues",
        "Source": "https://github.com/bioimage-io/pytorch-bioimage-io",
    },
)
