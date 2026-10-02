from datetime import datetime

from fastapi.testclient import TestClient
from runtime_support import service, settings

from apps.api.main import create_app
from apps.api.models import utcnow


def test_public_health_exposes_precise_server_time_without_changing_the_health_contract():
    before = utcnow()
    with TestClient(create_app(settings(), service())) as client:
        response = client.get("/healthz")
    after = utcnow()
    assert response.json() == {"status": "process_running", "deployment": "azure"}
    server_time = datetime.fromisoformat(response.headers["x-server-time"])
    assert before <= server_time <= after
