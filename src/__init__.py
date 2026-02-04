"""
Black-Scholes Option Pricing Platform - Core Package

A comprehensive quantitative finance platform for option pricing,
risk management, and portfolio analysis.
"""

__version__ = "2.1.0"

# Expose core packages for easy discovery
from . import config, database, api, pricing, ml, services, workers, streaming, aiops, shared, utils, audit, auth
