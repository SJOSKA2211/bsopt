from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.exceptions import (
    AuthenticationException,
    BaseAPIException,
    ConflictException,
    InternalServerException,
    NotFoundException,
    PermissionDeniedException,
    ServiceUnavailableException,
    ValidationException,
)
from src.api.main import api_exception_handler

# Create a minimal app for testing exceptions
app = FastAPI()


@app.exception_handler(BaseAPIException)
async def handler(request: Request, exc: BaseAPIException):
    return await api_exception_handler(request, exc)


@app.get("/raise/not-found")
async def raise_not_found():
    raise NotFoundException(message="Resource missing")


@app.get("/raise/validation")
async def raise_validation():
    raise ValidationException(message="Invalid data")


@app.get("/raise/permission")
async def raise_permission():
    raise PermissionDeniedException(message="No access")


@app.get("/raise/auth")
async def raise_auth():
    raise AuthenticationException(message="Login required")


@app.get("/raise/conflict")
async def raise_conflict():
    raise ConflictException(message="State conflict")


@app.get("/raise/service-unavailable")
async def raise_service_unavailable():
    raise ServiceUnavailableException(message="Down")


@app.get("/raise/internal")
async def raise_internal():
    raise InternalServerException(message="Oops")


client = TestClient(app)


def test_not_found_exception():
    response = client.get("/raise/not-found")
    assert response.status_code == 404
    data = response.json()
    assert data["error"] == "NotFound"
    assert data["message"] == "Resource missing"


def test_validation_exception():
    response = client.get("/raise/validation")
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "ValidationError"
    assert data["message"] == "Invalid data"


def test_permission_exception():
    response = client.get("/raise/permission")
    assert response.status_code == 403
    data = response.json()
    assert data["error"] == "PermissionDenied"
    assert data["message"] == "No access"


def test_auth_exception():
    response = client.get("/raise/auth")
    assert response.status_code == 401
    data = response.json()
    assert data["error"] == "AuthenticationFailed"
    assert data["message"] == "Login required"


def test_conflict_exception():
    response = client.get("/raise/conflict")
    assert response.status_code == 409
    data = response.json()
    assert data["error"] == "Conflict"
    assert data["message"] == "State conflict"


def test_service_unavailable_exception():
    response = client.get("/raise/service-unavailable")
    assert response.status_code == 503
    data = response.json()
    assert data["error"] == "ServiceUnavailable"
    assert data["message"] == "Down"


def test_internal_exception():
    response = client.get("/raise/internal")
    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "InternalServerError"
    assert data["message"] == "Oops"
