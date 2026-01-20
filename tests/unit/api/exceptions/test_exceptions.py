import pytest
from src.api.exceptions.exceptions import (
    BaseAPIException, 
    NotFoundException, 
    ValidationException, 
    PermissionDeniedException, 
    AuthenticationException, 
    ConflictException, 
    ServiceUnavailableException, 
    InternalServerException
)
from fastapi import status

def test_base_api_exception_defaults():
    exc = BaseAPIException()
    assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc.detail["error"] == "InternalServerError"

def test_base_api_exception_custom():
    exc = BaseAPIException(status_code=400, error="BadReq", message="Oops", details={"f": 1})
    assert exc.status_code == 400
    assert exc.detail["error"] == "BadReq"
    assert exc.detail["message"] == "Oops"
    assert exc.detail["details"] == {"f": 1}

def test_derived_exceptions():
    assert NotFoundException().status_code == 404
    assert ValidationException().status_code == 422
    assert PermissionDeniedException().status_code == 403
    assert AuthenticationException().status_code == 401
    assert ConflictException().status_code == 409
    assert ServiceUnavailableException().status_code == 503
    assert InternalServerException().status_code == 500
