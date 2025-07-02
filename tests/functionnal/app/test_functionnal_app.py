# import pytest
# from src.waterflow.app import app, FEATURES

# @pytest.fixture
# def client():
#     app.testing = True
#     return app.test_client()

# def test_home_page_get(client):
#     """La page d'accueil doit répondre 200 OK en GET"""
#     response = client.get('/')
#     assert response.status_code == 200
#     assert "Prédire la potabilité de l'eau" in response.data

# def test_prediction_post_valid_data(client):
#     """Test avec un POST complet et valide"""
#     valid_data = {feature: '7.0' for feature in FEATURES}
#     response = client.post('/', data=valid_data)
#     assert response.status_code == 200
#     assert 'Résultat de la prédiction' in response.data or 'Erreur' in response.data

# def test_prediction_post_missing_field(client):
#     """Envoi d'un formulaire incomplet : doit générer une erreur (handled gracefully)"""
#     incomplete_data = {feature: '7.0' for feature in FEATURES[:-1]}
#     response = client.post('/', data=incomplete_data)
#     assert response.status_code == 200
#     assert b'Erreur' in response.data
