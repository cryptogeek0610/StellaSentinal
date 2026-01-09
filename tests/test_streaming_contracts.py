from fastapi.testclient import TestClient

from device_anomaly.api.main import app


def test_streaming_status_endpoint():
    client = TestClient(app)
    response = client.get("/api/streaming/status")
    assert response.status_code == 200
    payload = response.json()
    assert "streaming_enabled" in payload
    assert "websocket_connections" in payload
