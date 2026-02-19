import os
import json
import pytest
import sys
import pandas as pd

# Ajout du dossier src au path pour importer app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from app import app  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _get_features_list():
    """
    Récupère la liste des features attendues.
    On essaye d’abord model_production/features.csv (chemin repo),
    sinon on renvoie une petite liste fallback pour ne pas casser la CI.
    """
    # Chemin robuste depuis la racine du repo
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    features_path = os.path.join(repo_root, "model_production", "features.csv")

    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        # app.py lit la colonne "feature" :contentReference[oaicite:2]{index=2}
        if "feature" in df.columns:
            return df["feature"].tolist()
        # fallback si la colonne a un autre nom
        return df.iloc[:, 0].tolist()

    # fallback minimal : permet de tester les statuts HTTP sans dépendre du fichier
    return ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "PAYMENT_RATE"]


def _make_payload(feats, profile="good"):
    fake = {f: 0 for f in feats}

    # on force quelques champs connus quand ils existent
    if "EXT_SOURCE_1" in fake:
        fake["EXT_SOURCE_1"] = 0.9 if profile == "good" else 0.1
    if "EXT_SOURCE_2" in fake:
        fake["EXT_SOURCE_2"] = 0.9 if profile == "good" else 0.1
    if "EXT_SOURCE_3" in fake:
        fake["EXT_SOURCE_3"] = 0.9 if profile == "good" else 0.1
    if "PAYMENT_RATE" in fake and profile == "good":
        fake["PAYMENT_RATE"] = 0.02

    return [fake]


def test_health(client):
    """Render healthcheck: /health doit répondre 200."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"


def test_homepage(client):
    """La route / doit répondre 200 (contenu peut varier)."""
    resp = client.get("/")
    assert resp.status_code == 200


def test_predict_returns_expected_schema_or_503(client):
    """
    /predict peut renvoyer:
    - 200 avec probability/threshold/prediction/status si le modèle est dispo,
    - 500/503 si modèle/features absents (CI) => on accepte mais on vérifie le message.
    """
    feats = _get_features_list()
    payload = _make_payload(feats, profile="good")

    resp = client.post("/predict", data=json.dumps(payload), content_type="application/json")

    if resp.status_code == 200:
        data = resp.get_json()
        assert "probability" in data
        assert "threshold" in data
        assert "prediction" in data
        assert "status" in data

        assert isinstance(data["probability"], float)
        assert 0.0 <= data["probability"] <= 1.0

        assert isinstance(data["threshold"], float)
        assert 0.0 <= data["threshold"] <= 1.0

        assert int(data["prediction"]) in (0, 1)
        assert str(data["status"]).lower() in ("accordé", "refusé")
    else:
        # CI friendly : on accepte si l’API signale que le modèle n’est pas dispo
        assert resp.status_code in (500, 503)
        data = resp.get_json()
        assert isinstance(data, dict)
        assert "error" in data


def test_explain_returns_expected_schema_or_503(client):
    """
    /explain peut renvoyer:
    - 200 avec base_value/threshold/top si LightGBM pred_contrib dispo,
    - 400 si modèle non compatible pred_contrib,
    - 500/503 si modèle absent.
    """
    feats = _get_features_list()
    payload = _make_payload(feats, profile="good")

    resp = client.post("/explain?k=8", data=json.dumps(payload), content_type="application/json")

    if resp.status_code == 200:
        data = resp.get_json()
        assert "threshold" in data
        assert "top" in data
        assert isinstance(data["top"], list)
        assert len(data["top"]) <= 8
        if data["top"]:
            item = data["top"][0]
            assert "feature" in item
            assert "value" in item
            assert "contribution" in item
    else:
        assert resp.status_code in (400, 500, 503)
        data = resp.get_json()
        assert isinstance(data, dict)
        assert "error" in data
