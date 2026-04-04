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
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    ParentBased,
    TraceIdRatioBased,
)
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer

from src.shared.config import settings

logger = structlog.get_logger(__name__)

_tracer: Tracer | None = None


def setup_tracing(
    service_name: str,
    service_version: str = "1.0.0",
    otlp_endpoint: str | None = None,
    sampling_ratio: float = 1.0,
    enable_console_export: bool = False,
) -> None:
    """
    Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: OTLP gRPC endpoint
        sampling_ratio: Fraction of traces to sample (0.0 to 1.0)
        enable_console_export: Whether to export traces to console
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
        sampler = ALWAYS_ON
    elif sampling_ratio <= 0.0:
        sampler = ALWAYS_OFF
    else:
        sampler = ParentBased(root=TraceIdRatioBased(sampling_ratio))

    provider = TracerProvider(resource=resource, sampler=sampler)

    try:
        if otlp_endpoint:
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
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls=",".join(excluded_urls),
        )
        logger.info("fastapi_instrumented")
    except ImportError:
        logger.warning("fastapi_instrumentation_not_available")
    except Exception as e:
        logger.warning("fastapi_instrumentation_failed", error=str(e))

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.info("httpx_instrumented")
    except ImportError:
        logger.warning("httpx_instrumentation_not_available")
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
        from opentelemetry.instrumentation.celery import CeleryInstrumentor
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


def get_tracer(name: str, version: str | None = None) -> trace.Tracer:
    """Get a tracer instance from the current provider."""
    return trace.get_tracer(name, version)


@contextmanager
def create_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
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
        _tracer = trace.get_tracer("bsopt")

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
    kind: SpanKind = SpanKind.INTERNAL,
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
