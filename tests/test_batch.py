from tests.test_api import VALID_PAYLOAD, client


def test_batch_one_customer(client):
    resp = client.post("/batch", json={"customers": [VALID_PAYLOAD]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 1


def test_batch_fifty_customers(client):
    resp = client.post("/batch", json={"customers": [VALID_PAYLOAD] * 50})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 50


def test_batch_empty_list_returns_422(client):
    resp = client.post("/batch", json={"customers": []})
    assert resp.status_code == 422


def test_batch_malformed_payload_returns_422(client):
    resp = client.post("/batch", json={"customers": [{"tenure": 12}]})
    assert resp.status_code == 422


def test_explain_returns_shap_values(client):
    resp = client.post("/explain", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert "churn_probability" in data
    assert isinstance(data["shap_values"], list)
    assert data["shap_values"]
