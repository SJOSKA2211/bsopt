"""
Distributed Tracing Configuration (OpenTelemetry)

Provides comprehensive distributed tracing across FastAPI, Celery, Ray, and other services.
Supports export to Grafana Tempo via OTLP gRPC.

Features:
- Auto-instrumentation for FastAPI, SQLAlchemy, Redis, httpx
- Ray task/actor tracing
- Custom span creation for critical paths
- Sampling strategies
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.trace.sampling import (
    AlwaysOffSampler,
    AlwaysOnSampler,
    ParentBased,
    TraceIdRatioBased,
)
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from src.shared.config import settings

logger = structlog.get_logger(__name__)

def setup_tracing(
    service_name: str,
    service_version: str = "1.0.0",
    otlp_endpoint: str | None = None,
    sampling_ratio: float = 1.0,
    enable_console_export: bool = False,
) -> None:
    """
    Configure OpenTelemetry tracing with OTLP export.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: OTLP endpoint (e.g., http://tempo:4317)
        sampling_ratio: Sampling ratio (0.0 to 1.0)
        enable_console_export: Enable console export for debugging
    """
    global _tracer

    env = settings.ENVIRONMENT
    enable_tracing = settings.ENABLE_TRACING

    if not enable_tracing:
        logger.info("tracing_disabled", service=service_name)
        return

    otlp_endpoint = otlp_endpoint or settings.OTEL_EXPORTER_OTLP_ENDPOINT

    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            "deployment.environment": env,
            "host.name": os.getenv("HOSTNAME", "unknown"),
            "process.pid": os.getpid(),
        }
    )

    if sampling_ratio >= 1.0:
        sampler = AlwaysOnSampler()
    elif sampling_ratio <= 0.0:
        sampler = AlwaysOffSampler()
    else:
        sampler = ParentBased(root=TraceIdRatioBased(sampling_ratio))

    provider = TracerProvider(resource=resource, sampler=sampler)

    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,
            timeout=30,
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info("otlp_exporter_configured", endpoint=otlp_endpoint)
    except Exception as e:
        logger.warning("otlp_export_failed", error=str(e))

    if enable_console_export or env in ("development", "dev"):
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("console_exporter_enabled")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name, service_version)

    logger.info(
        "tracing_initialized",
        service=service_name,
        version=service_version,
        sampling=sampling_ratio,
    )

def instrument_app(
    app: Any,
    excluded_urls: list[str] | None = None,
) -> None:
    """
    Instrument FastAPI application with OpenTelemetry.

    Args:
        app: FastAPI application instance
        excluded_urls: URLs to exclude from tracing
    """
    if not settings.ENABLE_TRACING:
        return

    excluded_urls = excluded_urls or [
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/openapi.json",
    ]

    try:
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls=excluded_urls,
            tracer_provider=trace.get_tracer_provider(),
        )
        logger.info("fastapi_instrumented")
    except Exception as e:
        logger.warning("fastapi_instrumentation_failed", error=str(e))

    try:
        HTTPXClientInstrumentor().instrument()
        logger.info("httpx_instrumented")
    except Exception as e:
        logger.warning("httpx_instrumentation_failed", error=str(e))

def instrument_database(engine: Any) -> None:
    """
    Instrument SQLAlchemy database engine.

    Args:
        engine: SQLAlchemy engine instance
    """
    if not settings.ENABLE_TRACING:
        return

    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument(engine=engine)
        logger.info("sqlalchemy_instrumented")
    except ImportError:
        logger.warning("sqlalchemy_instrumentation_not_available")
    except Exception as e:
        logger.warning("sqlalchemy_instrumentation_failed", error=str(e))

def instrument_redis(client: Any) -> None:
    """
    Instrument Redis client.

    Args:
        client: Redis client instance
    """
    if not settings.ENABLE_TRACING:
        return

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument(client=client)
        logger.info("redis_instrumented")
    except ImportError:
        logger.warning("redis_instrumentation_not_available")
    except Exception as e:
        logger.warning("redis_instrumentation_failed", error=str(e))

def instrument_celery() -> None:
    """Instrument Celery worker with OpenTelemetry."""
    if not settings.ENABLE_TRACING:
        return

    try:
        CeleryInstrumentor().instrument()
        logger.info("celery_instrumented")
    except ImportError:
        logger.warning("celery_instrumentation_not_available")
    except Exception as e:
        logger.warning("celery_instrumentation_failed", error=str(e))

def instrument_ray() -> None:
    """Instrument Ray tasks and actors."""
    if not settings.ENABLE_TRACING:
        return

    try:
        from opentelemetry.instrumentation.ray import RayInstrumentor

        RayInstrumentor().instrument()
        logger.info("ray_instrumented")
    except ImportError:
        logger.warning("ray_instrumentation_not_available")
    except Exception as e:
        logger.warning("ray_instrumentation_failed", error=str(e))

@contextmanager
def create_span(
    name: str,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
):
    """
    Create a new span for tracing.

    Args:
        name: Name of the span
        kind: Kind of span (internal, server, client, producer, consumer)
        attributes: Optional attributes to add to the span

    Usage:
        with create_span("my_operation", attributes={"key": "value"}) as span:
            # do work
            span.set_attribute("result", "success")
    """
    global _tracer

    if _tracer is None:
        _tracer = trace.get_tracer("equaflow")

    with _tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

def trace_function(
    name: str | None = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator to trace a function.

    Args:
        name: Optional name for the span (defaults to function name)
        kind: Kind of span
        attributes: Optional attributes to add

    Usage:
        @trace_function("my_function", attributes={"version": "1.0"})
        def my_function(x, y):
            return x + y
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with create_span(span_name, kind=kind, attributes=attributes) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with create_span(span_name, kind=kind, attributes=attributes) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                return await func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

def inject_trace_context() -> dict[str, str]:
    """
    Inject current trace context into a dictionary for propagation.

    Returns:
        Dictionary with trace context headers
    """
    carrier: dict[str, str] = {}
    _propagator.inject(carrier)
    return carrier

def extract_trace_context(carrier: dict[str, str]) -> Any:
    """
    Extract trace context from a dictionary.

    Args:
        carrier: Dictionary with trace context headers

    Returns:
        Extracted context
    """
    return _propagator.extract(carrier)

def get_current_span() -> Span | None:
    """Get the current active span."""
    return trace.get_current_span()

def add_span_attributes(attributes: dict[str, Any]) -> None:
    """Add attributes to the current span."""
    span = get_current_span()
    if span and span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)

class TraceContextManager:
    """
    Context manager for handling distributed trace context.

    Usage:
        async with TraceContextManager() as ctx:
            # trace context is available in ctx.context
            await some_async_operation()
    """

    def __init__(self, carrier: dict[str, str] | None = None):
        self.carrier = carrier or {}
        self.context = None

    async def __aenter__(self):
        self.context = extract_trace_context(self.carrier)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            span = get_current_span()
            if span:
                span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                span.record_exception(exc_val)
        return False

    def __enter__(self):
        self.context = extract_trace_context(self.carrier)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            span = get_current_span()
            if span:
                span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                span.record_exception(exc_val)
        return False

def shutdown_tracing() -> None:
    """Shutdown the tracer provider and flush pending spans."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    logger.info("tracing_shutdown")

if __name__ == "__main__":
    import sys

    service = sys.argv[1] if len(sys.argv) > 1 else "test-service"

    setup_tracing(
        service_name=service,
        enable_console_export=True,
        sampling_ratio=1.0,
    )

    @trace_function("example_operation")
    def example_function(x: int, y: int) -> int:
        return x + y

    @trace_function("example_async_operation", attributes={"async": True})
    async def example_async_function(x: int) -> int:
        import asyncio

        await asyncio.sleep(0.1)
        return x * 2

    with create_span("synchronous_work") as span:
        result = example_function(1, 2)
        span.set_attribute("result", result)
        print(f"Synchronous result: {result}")

    import asyncio

    asyncio.run(example_async_function(5))

    print("Tracing example complete")
    shutdown_tracing()
