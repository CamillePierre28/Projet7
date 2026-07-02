# Importe TestClient pour tester l'API FastAPI sans lancer un vrai serveur.
from fastapi.testclient import TestClient

# Importe l'application FastAPI a tester.
from api.main import app

# Cree un client de test capable d'appeler les routes de l'API.
client = TestClient(app)


# Teste la route GET /health.
def test_health_endpoint():
    # Appelle la route de sante de l'API.
    response = client.get("/health")

    # Verifie que la route repond avec un statut HTTP 200.
    assert response.status_code == 200

    # Convertit la reponse JSON en dictionnaire Python.
    data = response.json()

    # Verifie que le statut retourne est ok.
    assert data["status"] == "ok"
    # Verifie que l'information rag_loaded est presente.
    assert "rag_loaded" in data


# Teste la route POST /ask avec une question valide.
def test_ask_valid_question():
    # Envoie une question au chatbot via l'API.
    response = client.post(
        # Route appelee.
        "/ask",
        # Corps JSON envoye a l'API.
        json={
            # Question utilisateur de test.
            "question": "Quels événements culturels sont prévus à Toulouse ?"
        },
    )

    # Si le RAG n'est pas charge dans l'environnement de test,
    # on ne fait pas echouer le test API.
    if response.status_code == 503:
        # Arrete le test car l'API a correctement signale que le RAG est indisponible.
        return

    # Verifie que la reponse est un succes.
    assert response.status_code == 200

    # Convertit la reponse JSON en dictionnaire Python.
    data = response.json()

    # Verifie que la question est presente dans la reponse.
    assert "question" in data
    # Verifie que la reponse generee est presente.
    assert "answer" in data
    # Verifie que les sources sont presentes.
    assert "sources" in data
    # Verifie que l'evaluation automatique est presente.
    assert "evaluation" in data

    # Recupere la partie evaluation de la reponse.
    evaluation = data["evaluation"]

    # Verifie que le nombre de sources est present.
    assert "sources_count" in evaluation
    # Verifie que la ville attendue est presente.
    assert "expected_city" in evaluation
    # Verifie que le booleen city_ok est present.
    assert "city_ok" in evaluation
    # Verifie que le mot-cle attendu est present.
    assert "expected_keyword" in evaluation
    # Verifie que le booleen keyword_ok est present.
    assert "keyword_ok" in evaluation
    # Verifie que l'information requires_future est presente.
    assert "requires_future" in evaluation
    # Verifie que le booleen future_ok est present.
    assert "future_ok" in evaluation
    # Verifie que la classification finale est presente.
    assert "classification" in evaluation

    # Verifie que sources_count est bien un entier.
    assert isinstance(evaluation["sources_count"], int)
    # Verifie que city_ok est bien un booleen.
    assert isinstance(evaluation["city_ok"], bool)
    # Verifie que keyword_ok est bien un booleen.
    assert isinstance(evaluation["keyword_ok"], bool)
    # Verifie que requires_future est bien un booleen.
    assert isinstance(evaluation["requires_future"], bool)
    # Verifie que future_ok est bien un booleen.
    assert isinstance(evaluation["future_ok"], bool)
    # Verifie que la classification fait partie des valeurs autorisees.
    assert evaluation["classification"] in [
        # Reponse consideree bonne.
        "correcte",
        # Reponse consideree partiellement bonne.
        "partiellement correcte",
        # Reponse consideree mauvaise.
        "incorrecte",
    ]


# Teste que l'API refuse une question vide.
def test_ask_empty_question():
    # Envoie une question vide.
    response = client.post(
        # Route appelee.
        "/ask",
        # Corps JSON avec une question vide.
        json={"question": ""},
    )

    # FastAPI/Pydantic doit refuser car min_length=1.
    assert response.status_code == 422


# Teste que l'API refuse une question composee uniquement d'espaces.
def test_ask_spaces_only_question():
    # Envoie une question contenant seulement des espaces.
    response = client.post(
        # Route appelee.
        "/ask",
        # Corps JSON de test.
        json={"question": "   "},
    )

    # La validation Pydantic accepte les espaces,
    # mais la logique metier rejette la question apres strip().
    if response.status_code == 503:
        # Arrete le test si le RAG n'est pas charge dans cet environnement.
        return

    # Verifie que l'API renvoie une erreur de mauvaise requete.
    assert response.status_code == 400

    # Convertit la reponse JSON en dictionnaire Python.
    data = response.json()

    # Verifie que le message d'erreur est celui attendu.
    assert data["detail"] == "La question ne peut pas être vide."


# Teste que l'API refuse un payload sans champ question.
def test_ask_missing_question():
    # Envoie un JSON avec un mauvais nom de champ.
    response = client.post(
        # Route appelee.
        "/ask",
        # Corps JSON incomplet.
        json={"wrong_field": "test"},
    )

    # FastAPI doit signaler une erreur de validation.
    assert response.status_code == 422


# Teste que l'API refuse un payload vide.
def test_ask_empty_payload():
    # Envoie un dictionnaire vide.
    response = client.post(
        # Route appelee.
        "/ask",
        # Corps JSON vide.
        json={},
    )

    # FastAPI doit signaler une erreur de validation.
    assert response.status_code == 422


# Teste que la route /rebuild refuse une requete sans cle API.
def test_rebuild_without_key_is_unauthorized(monkeypatch):
    # Force la cle attendue par l'API pendant ce test.
    monkeypatch.setattr("api.main.REBUILD_API_KEY", "secret")

    # Appelle /rebuild sans fournir le header X-API-Key.
    response = client.post("/rebuild")

    # Verifie que l'API refuse la requete.
    assert response.status_code == 401