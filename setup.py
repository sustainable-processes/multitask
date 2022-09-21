#!/usr/bin/env python

import os
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

with open("requirements.txt", "r") as f:
    reqs = f.readlines()

setup(
    name="multitask",
    version="0.1.0",
    packages=find_packages(exclude=["figures", "data", "tests", "nbs"]),
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    # keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.8",
    setup_requires=["wheel"],
    install_requires=reqs,
)
