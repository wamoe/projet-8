from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- CONFIGURATION DES CHEMINS (ROBUSTE) ---
current_dir = os.path.dirname(os.path.abspath(__file__))   # dossier src/
root_dir = os.path.dirname(current_dir)                   # racine projet

MODEL_PATH = os.path.join(root_dir, "model_production", "model.pkl")
FEATURES_PATH = os.path.join(root_dir, "model_production", "features.csv")
THRESHOLD_PATH = os.path.join(root_dir, "model_production", "threshold.txt")

print(f"Chemin du modèle détecté : {MODEL_PATH}")
print("App boot...")

model = None
expected_cols = None


def ensure_loaded():
    """Charge features + modèle à la demande (évite crash au boot / cold start)."""
    global model, expected_cols

    if expected_cols is None:
        df_feat = pd.read_csv(FEATURES_PATH)
        expected_cols = (
            df_feat["feature"].tolist()
            if "feature" in df_feat.columns
            else df_feat.iloc[:, 0].tolist()
        )

    if model is None:
        print("Chargement du modèle (lazy)...")
        model = joblib.load(MODEL_PATH)
        print("Modèle chargé.")

    return model, expected_cols


def load_threshold(default=0.5) -> float:
    thr = default
    try:
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, "r") as f:
                thr = float(f.read().strip())
    except Exception:
        thr = default
    return float(thr)


# -------------------------
# ROUTES "LÉGÈRES" (Render)
# -------------------------
@app.route("/health", methods=["GET", "HEAD"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET", "HEAD"])
def home():
    return "ok", 200


# -------------------------
# API
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        model, expected_cols = ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Model not available: {e}"}), 503

    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        df = df.reindex(columns=expected_cols, fill_value=0)

        y_proba = float(model.predict_proba(df)[:, 1][0])
        thr = load_threshold(default=0.5)

        decision = 1 if y_proba >= thr else 0
        status = "Refusé" if decision == 1 else "Accordé"

        return jsonify({
            "probability": y_proba,
            "threshold": float(thr),
            "prediction": int(decision),
            "status": status
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    try:
        model, expected_cols = ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Model not available: {e}"}), 503

    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        df = df.reindex(columns=expected_cols, fill_value=0)

        # récupérer le booster LightGBM (si dispo)
        booster = None
        if hasattr(model, "booster_"):
            booster = model.booster_
        elif hasattr(model, "predict") and hasattr(model.predict, "__code__") and "pred_contrib" in model.predict.__code__.co_varnames:
            booster = model
        else:
            return jsonify({
                "error": "Explication indisponible: modèle non compatible pred_contrib (LightGBM requis)."
            }), 400

        contrib = booster.predict(df, pred_contrib=True)
        contrib = np.asarray(contrib)

        vals = contrib[0, :-1].astype(float)
        base_val = float(contrib[0, -1])

        k = int(request.args.get("k", 12))
        k = max(1, min(k, len(expected_cols)))
        idx = np.argsort(np.abs(vals))[::-1][:k]

        top = []
        for j in idx:
            top.append({
                "feature": expected_cols[j],
                "value": float(df.iloc[0, j]),
                "contribution": float(vals[j]),
            })

        thr = load_threshold(default=0.5)

        return jsonify({
            "base_value": base_val,
            "threshold": float(thr),
            "top": top
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500