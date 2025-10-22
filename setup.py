import importlib.util

if importlib.util.find_spec("setuptools_scm") is None:
    raise ImportError(
        "setuptools-scm is not installed. Install it by `pip3 install setuptools-scm`"
    )

import os
import subprocess
from os import path

from setuptools import find_packages, setup
from setuptools_scm.version import get_local_dirty_tag


PACKAGE_NAME = "cache-dit"


def is_git_directory(path="."):
    return (
        subprocess.call(
            ["git", "-C", path, "status"],
            stderr=subprocess.STDOUT,
            stdout=open(os.devnull, "w"),
        )
        == 0
    )


def my_local_scheme(version):
    # The following is used to build release packages.
    # Users should never use it.
    local_version = os.getenv("CACHE_DIT_BUILD_LOCAL_VERSION")
    if local_version is None:
        return get_local_dirty_tag(version)
    return f"+{local_version}"


def fetch_core_requirements():
    with open("requirements/core.txt") as f:
        return f.read().strip().split("\n")


def fetch_quantization_requirements():
    with open("requirements/quantization.txt") as f:
        return f.read().strip().split("\n")


def fetch_metrics_requirements():
    with open("requirements/metrics.txt") as f:
        return f.read().strip().split("\n")


setup(
    name=PACKAGE_NAME,
    description="A Unified, Flexible and Training-free Cache Acceleration Framework for 🤗Diffusers.",
    author="DefTruth, vipshop.com",
    use_scm_version={
        "write_to": path.join("src", "cache_dit", "_version.py"),
        "local_scheme": my_local_scheme,
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "cache-dit-metrics-cli = cache_dit.metrics:main",
        ],
    },
    # pip3 install cache-dit, installs core and quantization requirements by default
    install_requires=fetch_core_requirements()
    + fetch_quantization_requirements(),
    extras_require={
        # pip3 install 'cache-dit[core]'
        "core": fetch_core_requirements(),
        # pip3 install 'cache-dit[quantization]'
        "quantization": fetch_quantization_requirements(),
        # pip3 install 'cache-dit[metrics]
        "metrics": fetch_metrics_requirements(),
        # pip3 install 'cache-dit[dev]'
        "dev": [
            "pre-commit",
            "pytest>=7.0.0,<8.0.0",
            "pytest-html",
            "expecttest",
            "hypothesis",
            "accelerate",
            "peft",
            "protobuf",
            "sentencepiece",
            "opencv-python-headless",
            "ftfy",
            "scikit-image",
        ],
        # pip3 install 'cache-dit[all]'
        "all": fetch_metrics_requirements(),
    },
)
