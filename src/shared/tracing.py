"""
Distributed Tracing Configuration (OpenTelemetry)
=================================================

Provides a centralized OpenTelemetry configuration for distributed tracing
across FastAPI, Celery, and other services.
"""

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


def setup_tracing(service_name: str):
    """
    Configure OpenTelemetry tracing.
    """
    # Check if tracing is enabled via env var
    if os.getenv("ENABLE_TRACING", "false").lower() != "true":
        return

    resource = Resource.create({
        "service.name": service_name,
        "deployment.environment": os.getenv("ENV", "production"),
    })

    provider = TracerProvider(resource=resource)
    
    # Export to Jaeger/Tempo via OTLP
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    
    processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(processor)

    # Optional: Console exporter for local debugging
    if os.getenv("ENV") == "development":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

def instrument_app(app):
    """Instrument FastAPI application."""
    if os.getenv("ENABLE_TRACING", "false").lower() == "true":
        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        # Note: SQLAlchemy and Redis instrumentation should be called where the clients are created
        # or globally here if possible/safe.
        # RedisInstrumentor().instrument() 
        # SQLAlchemyInstrumentor().instrument(engine=engine) 

def instrument_celery():
    """Instrument Celery worker."""
    if os.getenv("ENABLE_TRACING", "false").lower() == "true":
        CeleryInstrumentor().instrument()
