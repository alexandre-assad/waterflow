import pytest
from waterflow.app import app, FEATURES


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


def test_home_page_get(client):
    """La page d'accueil doit répondre 200 OK en GET"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Prédire la potabilité de l'eau" in response.data.decode()


def test_prediction_post_valid_data(client):
    """Test avec un POST complet et valide"""
    valid_data = {feature: "7.0" for feature in FEATURES}
    response = client.post("/", data=valid_data)
    assert response.status_code == 200
    assert "Résultat de la prédiction" in response.data.decode()
