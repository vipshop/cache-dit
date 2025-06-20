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


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


setup(
    name=PACKAGE_NAME,
    description="🤗 CacheDiT: A Training-free and Easy-to-use Cache Acceleration Toolbox for Diffusion Transformers",
    author="vipshop.com",
    use_scm_version={
        "write_to": path.join("src", "cache_dit", "_version.py"),
        "local_scheme": my_local_scheme,
    },
    package_dir={
        "": "src",
    },
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=fetch_requirements(),
    extras_require={
        "all": [],
        # dev dependencies. Install them by `pip3 install 'cache-dit[dev]'`
        "dev": [
            "pre-commit",
            "pytest>=7.0.0,<8.0.0",
            "pytest-html",
            "expecttest",
            "hypothesis",
            "transformers",
            # "diffusers @ git+https://github.com/huggingface/diffusers",  # wan currently requires installing from source
            "diffusers",
            "accelerate",
            "peft",
            "protobuf",
            "sentencepiece",
            "opencv-python-headless",
            "ftfy",
        ],
    },
)
