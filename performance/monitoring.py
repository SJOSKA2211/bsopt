"""
Performance Monitoring
======================

Implements Prometheus metrics and system resource tracking.
"""

import time
import logging
import psutil
import functools
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = Info = None
    CollectorRegistry = None

logger = logging.getLogger(__name__)

if PROMETHEUS_AVAILABLE:
    registry = CollectorRegistry()
    
    # API Metrics
    api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'], registry=registry)
    api_request_duration_seconds = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'], registry=registry)
    api_requests_in_progress = Gauge('api_requests_in_progress', 'Active API requests', ['method', 'endpoint'], registry=registry)

    # Pricing Metrics
    pricing_calculations_total = Counter('pricing_calculations_total', 'Total pricing calculations', ['method', 'option_type'], registry=registry)
    pricing_calculation_duration_seconds = Histogram('pricing_calculation_duration_seconds', 'Pricing duration', ['method', 'option_type'], registry=registry)
    implied_vol_calculations_total = Counter('implied_vol_calculations_total', 'Total IV calculations', ['method', 'status'], registry=registry)

    # System Metrics
    system_cpu_usage_percent = Gauge('system_cpu_usage_percent', 'System CPU usage', registry=registry)
    system_memory_usage_percent = Gauge('system_memory_usage_percent', 'System memory usage', registry=registry)
    
    # Database Metrics
    db_connection_pool_size = Gauge('db_connection_pool_size', 'DB pool size', registry=registry)
    db_connection_pool_checked_out = Gauge('db_connection_pool_checked_out', 'DB connections in use', registry=registry)

    app_info = Info('app_info', 'Application information', registry=registry)
    app_info.info({'version': '3.0.0', 'name': 'bsopt'})

def update_system_metrics():
    if not PROMETHEUS_AVAILABLE: return
    try:
        system_cpu_usage_percent.set(psutil.cpu_percent())
        system_memory_usage_percent.set(psutil.virtual_memory().percent)
    except Exception as e:
        logger.error(f"Failed to update system metrics: {e}")

def update_db_pool_metrics(engine):
    if not PROMETHEUS_AVAILABLE: return
    try:
        pool = engine.pool
        db_connection_pool_size.set(pool.size())
        db_connection_pool_checked_out.set(pool.checkedout())
    except Exception as e:
        logger.error(f"Failed to update DB pool metrics: {e}")

def get_metrics() -> bytes:
    return generate_latest(registry) if PROMETHEUS_AVAILABLE else b""

def get_content_type() -> str:
    return CONTENT_TYPE_LATEST if PROMETHEUS_AVAILABLE else "text/plain"