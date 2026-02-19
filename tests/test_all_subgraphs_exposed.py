from fastapi.testclient import TestClient


def test_pricing_subgraph_exposed():
    from src.pricing.main import app

    client = TestClient(app)
    response = client.get("/graphql")
    assert response.status_code == 200  # GraphiQL interface


def test_ml_subgraph_exposed():
    from src.ml.main import app

    client = TestClient(app)
    response = client.get("/graphql")
    assert response.status_code == 200


def test_portfolio_subgraph_exposed():
    from src.portfolio.main import app

    client = TestClient(app)
    response = client.get("/graphql")
    assert response.status_code == 200


def test_streaming_subgraph_exposed():
    from src.streaming.main import app

    client = TestClient(app)
    response = client.get("/graphql")
    assert response.status_code == 200
