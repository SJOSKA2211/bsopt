
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.middleware.security import InputSanitizationMiddleware

app = FastAPI()
app.add_middleware(InputSanitizationMiddleware, log_suspicious=True)

@app.get("/test")
def endpoint():
    return {"message": "ok"}

client = TestClient(app)

def test_input_sanitization_blocks_xss():
    # It SHOULD block, so we expect 400
    response = client.get("/test?q=<script>alert(1)</script>")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid input detected"}

def test_input_sanitization_allows_safe_input():
    response = client.get("/test?q=hello")
    assert response.status_code == 200
    assert response.json() == {"message": "ok"}
