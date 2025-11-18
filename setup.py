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


setup(
    name=PACKAGE_NAME,
    description="A Unified and Flexible Inference Engine with Hybrid Cache Acceleration and Parallelism for 🤗DiTs.",
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
)
