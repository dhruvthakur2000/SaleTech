import pytest
from src.saletech.utils.errors import (
    SaleTechException,
    SessionNotFoundError,
    ValidationError,
    AuthorizationError
    )

def test_saletech_exception_to_dict():
    exc = SaleTechException("Test Error", "TEST_ERROR", 500, {"foo": "bar"})
    d= exc.to_dict()
    assert d["message"] == "Test Error"
    assert d["error_code"] == "TEST_ERROR"
    assert d["status_code"] == 500
    assert d["context"] == {"foo": "bar"}    

def test_session_not_found_error():
    exc = SessionNotFoundError("abc123")
    assert exc.error_code == "SESSION_NOT_FOUND"
    assert exc.status_code == 404
    assert exc.context["session_id"] == "abc123"

