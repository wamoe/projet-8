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

print(f"Chemin du modèle détecté : {MODEL_PATH}")

# --- CHARGEMENT DU MODÈLE ---
print("Chargement du modèle...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Le fichier n'existe pas : {MODEL_PATH}")
        
    model = joblib.load(MODEL_PATH)
    
    # Chargement des features attendues
    expected_features = pd.read_csv(FEATURES_PATH)
    expected_cols = expected_features.iloc[:, 0].tolist()
    
    print(" Modèle chargé avec succès.")
except Exception as e:
    print(f" ERREUR CRITIQUE lors du chargement du modèle : {e}")
    model = None

@app.route('/')
def home():
    status_model = " Chargé" if model else " Non chargé (Voir logs terminal)"
    return f"<h1>API de Scoring Crédit</h1><p>État du modèle : {status_model}</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Le modèle n\'est pas chargé. Vérifiez les logs du serveur.'}), 500
    
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
        optimal_threshold = 0.45 
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
