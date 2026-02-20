from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- CONFIGURATION DES CHEMINS (ROBUSTE) ---
# On récupère le dossier où se trouve ce fichier app.py (c'est-à-dire le dossier 'src')
current_dir = os.path.dirname(os.path.abspath(__file__))

# On remonte d'un cran pour aller à la racine du projet
root_dir = os.path.dirname(current_dir)

# On construit les chemins absolus vers le modèle et les features
MODEL_PATH = os.path.join(root_dir, 'model_production', 'model.pkl')
FEATURES_PATH = os.path.join(root_dir, 'model_production', 'features.csv')
THRESHOLD_PATH = os.path.join(root_dir, "model_production", "threshold.txt")

print(f"Chemin du modèle détecté : {MODEL_PATH}")

# --- CHARGEMENT DU MODÈLE ---
print("Chargement du modèle...")
model = None
expected_cols = None

def ensure_loaded():
    global model, expected_cols
    if expected_cols is None:
        df_feat = pd.read_csv(FEATURES_PATH)
        expected_cols = df_feat["feature"].tolist() if "feature" in df_feat.columns else df_feat.iloc[:, 0].tolist()
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





@app.route("/health", methods=["GET", "HEAD"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/", methods=["GET", "HEAD"])
def home():
    return "ok", 200


@app.route('/explain', methods=['POST'])
def explain():
    """
    Explication locale : contributions type SHAP via LightGBM pred_contrib
    (robuste, évite les soucis de dépendances shap/numba en prod).
    """
    try:
        model, expected_cols = ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Model not available: {e}"}), 503

    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        # mêmes colonnes qu’à l’entraînement
        df = df.reindex(columns=expected_cols, fill_value=0)

        # récupérer le booster LightGBM
        booster = None
        if hasattr(model, "booster_"):           # LGBMClassifier / LGBMRegressor
            booster = model.booster_
        elif hasattr(model, "predict") and "pred_contrib" in getattr(model.predict, "__code__", ()).co_varnames:
            booster = model  # cas rare : modèle directement booster
        else:
            # fallback : pas un modèle LightGBM compatible pred_contrib
            return jsonify({
                "error": "Explication indisponible: modèle non compatible pred_contrib (LightGBM requis)."
            }), 400

        contrib = booster.predict(df, pred_contrib=True)
        contrib = np.asarray(contrib)

        # contrib shape = (n, n_features + 1) ; dernier = base_value (bias)
        vals = contrib[0, :-1].astype(float)
        base_val = float(contrib[0, -1])

        # top-k (par abs)
        k = int(request.args.get("k", 12))
        k = max(1, min(k, len(expected_cols)))
        idx = np.argsort(np.abs(vals))[::-1][:k]

        top = []
        for j in idx:
            feat = expected_cols[j]
            top.append({
                "feature": feat,
                "value": float(df.iloc[0, j]),
                "contribution": float(vals[j])
            })

        thr = load_threshold(default=0.5)

        return jsonify({
            "base_value": base_val,
            "threshold": float(thr),
            "top": top
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])

def predict():
    model, expected_cols = ensure_loaded()

    try:
        model, expected_cols = ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Model not available: {e}"}), 503
    
    try:
        # 1. Récupération des données JSON
        data = request.get_json()
        
        # 2. Transformation en DataFrame
        df = pd.DataFrame(data)
        
        # 3. Réorganisation des colonnes (Match exact avec l'entraînement)
        # On ne garde que les colonnes attendues, dans le bon ordre
        # Si une colonne manque, on remplit par 0 (ou NaN selon le besoin, ici 0 pour la robustesse)
        df = df.reindex(columns=expected_cols, fill_value=0)

        # 4. Prédiction
        y_proba = model.predict_proba(df)[:, 1][0]
        
        # 5. Interprétation
        # Seuil par défaut (fallback)
        optimal_threshold = load_threshold(default=0.5)


        print(f"Seuil utilisé par l'API : {optimal_threshold}") 

        decision = 1 if y_proba >= optimal_threshold else 0
        
        return jsonify({
            'probability': float(y_proba),
            'threshold': optimal_threshold,
            'prediction': int(decision),
            'status': 'Refusé' if decision == 1 else 'Accordé'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
