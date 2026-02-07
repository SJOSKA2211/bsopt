"""
Black-Scholes Option Pricing Platform - Core Package

A comprehensive quantitative finance platform for option pricing,
risk management, and portfolio analysis.
"""

__version__ = "2.1.0"

# Expose core packages for easy discovery
from . import (
    aiops,
    api,
    audit,
    auth,
    config,
    database,
    ml,
    pricing,
    services,
    shared,
    streaming,
    utils,
    workers,
)
