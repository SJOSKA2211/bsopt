"""
Setup configuration for Black-Scholes Option Pricing Platform CLI

Installs the 'bsopt' command-line tool.
"""

from setuptools import find_packages, setup

setup(
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    entry_points={
        "console_scripts": [
            "bsopt=cli_complete:main",
        ],
    },
)
