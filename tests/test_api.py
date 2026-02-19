import os
import sys
import pytest

# Import app.py depuis src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from app import app  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    # Important: si Flask propague les exceptions en mode testing, on évite
    # d'appeler les endpoints susceptibles de lever pendant l'exécution.
    app.config["PROPAGATE_EXCEPTIONS"] = False
    with app.test_client() as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"


def test_root_ok(client):
    r = client.get("/")
    assert r.status_code == 200
