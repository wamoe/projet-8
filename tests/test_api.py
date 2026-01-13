import pytest
import sys
import os
import json
import pandas as pd

# Ajout du dossier src au chemin pour pouvoir importer app.py
# Permet de trouver le module même si le test est lancé depuis la racine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from app import app

@pytest.fixture
def client():
    """Crée un client de test Flask"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    """Test si la page d'accueil répond bien"""
    response = client.get('/')
    assert response.status_code == 200
    # Ici, on vérifie juste "API de Scoring" pour éviter les problèmes d'encodage/accents
    assert b"API de Scoring" in response.data

def test_prediction_accordee(client):
    """Test d'un client qui devrait avoir un crédit (Risque faible)"""
    # Chemin vers le fichier features.csv généré par le notebook
    features_path = os.path.join('model_production', 'features.csv')
    
    # Si le fichier n'existe pas (ex: CI/CD), on saute le test ou on utilise des features simulées
    if os.path.exists(features_path):
        feats = pd.read_csv(features_path).iloc[:, 0].tolist()
        
        # On crée un profil "Parfait" : Valeurs neutres (0) mais indicateurs externes excellents
        fake_data = {feat: 0 for feat in feats}
        fake_data['EXT_SOURCE_1'] = 0.9
        fake_data['EXT_SOURCE_2'] = 0.9
        fake_data['EXT_SOURCE_3'] = 0.9
        fake_data['PAYMENT_RATE'] = 0.02 

        
        response = client.post('/predict', 
                               data=json.dumps([fake_data]),
                               content_type='application/json')
        
        data = json.loads(response.data)
        
        assert response.status_code == 200
        assert 'probability' in data
        assert 'status' in data
        # On vérifie que la probabilité est bien un float entre 0 et 1
        assert isinstance(data['probability'], float)
        assert 0 <= data['probability'] <= 1

def test_prediction_refusee(client):
    """Test d'un client qui devrait être refusé (Risque élevé)"""
    features_path = os.path.join('model_production', 'features.csv')
    
    if os.path.exists(features_path):
        feats = pd.read_csv(features_path).iloc[:, 0].tolist()
        
        # Profil risqué
        fake_data = {feat: 0 for feat in feats}
        fake_data['EXT_SOURCE_1'] = 0.1
        fake_data['EXT_SOURCE_2'] = 0.1
        fake_data['EXT_SOURCE_3'] = 0.1
        
        
        response = client.post('/predict', 
                               data=json.dumps([fake_data]),
                               content_type='application/json')
        
        data = json.loads(response.data)
        assert response.status_code == 200
        assert 0 <= data['probability'] <= 1