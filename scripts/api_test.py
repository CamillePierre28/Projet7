# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe json pour afficher proprement les reponses JSON.
import json

# Importe requests pour envoyer des requetes HTTP a l'API locale.
import requests

# Definit l'adresse de base de l'API locale.
BASE_URL = "http://127.0.0.1:8000"


# Affiche un separateur lisible avant chaque test manuel.
def print_separator(title: str) -> None:
    # Affiche une ligne vide puis une grande ligne de signes egal.
    print("\n" + "=" * 70)
    # Affiche le titre du test.
    print(title)
    # Affiche une deuxieme ligne de signes egal.
    print("=" * 70)


# Teste la route GET /health.
def test_health() -> None:
    # Affiche le nom du test.
    print_separator("TEST : GET /health")

    # Envoie une requete GET vers la route /health.
    response = requests.get(
        # URL complete de la route.
        f"{BASE_URL}/health",
        # Temps maximum d'attente de la reponse.
        timeout=10,
    )

    # Affiche le code HTTP retourne par l'API.
    print(f"Status code : {response.status_code}")
    # Affiche la reponse JSON formatee.
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))


# Teste la route POST /ask avec une question valide.
def test_ask_valid_question() -> None:
    # Affiche le nom du test.
    print_separator("TEST : POST /ask (question valide)")

    # Prepare le corps JSON envoye a l'API.
    payload = {
        # Question posee au chatbot.
        "question": "Quels événements culturels sont prévus à Toulouse ?"
    }

    # Envoie une requete POST vers /ask.
    response = requests.post(
        # URL complete de la route.
        f"{BASE_URL}/ask",
        # Corps JSON de la requete.
        json=payload,
        # Timeout plus long car le RAG peut appeler Mistral.
        timeout=120,
    )

    # Affiche le code HTTP retourne.
    print(f"Status code : {response.status_code}")

    # Convertit la reponse JSON en dictionnaire Python.
    data = response.json()

    # Affiche un titre pour la reponse generee.
    print("\nReponse generee :")
    # Affiche le texte de la reponse.
    print(data["answer"])

    # Affiche le nombre de sources renvoyees.
    print("\nNombre de sources :", len(data["sources"]))

    # Affiche un titre pour l'evaluation.
    print("\nEvaluation automatique :")
    # Affiche l'evaluation JSON de facon lisible.
    print(json.dumps(data["evaluation"], indent=4, ensure_ascii=False))


# Teste la route /ask avec une question vide.
def test_ask_empty_question() -> None:
    # Affiche le nom du test.
    print_separator("TEST : POST /ask (question vide)")

    # Prepare un payload avec une question vide.
    payload = {"question": ""}

    # Envoie la requete POST vers /ask.
    response = requests.post(
        # URL complete de la route.
        f"{BASE_URL}/ask",
        # Corps JSON de test.
        json=payload,
        # Timeout court car aucune generation n'est attendue.
        timeout=10,
    )

    # Affiche le code HTTP retourne.
    print(f"Status code : {response.status_code}")
    # Affiche le texte brut de la reponse.
    print(response.text)


# Teste la route /ask avec une question composee uniquement d'espaces.
def test_ask_spaces_only_question() -> None:
    # Affiche le nom du test.
    print_separator("TEST : POST /ask (espaces uniquement)")

    # Envoie une question qui contient seulement des espaces.
    response = requests.post(
        # URL complete de la route.
        f"{BASE_URL}/ask",
        # Corps JSON avec une fausse question.
        json={"question": "     "},
        # Timeout court car la requete doit etre refusee rapidement.
        timeout=10,
    )

    # Affiche le code HTTP retourne.
    print(f"Status code : {response.status_code}")
    # Affiche le texte brut de la reponse.
    print(response.text)


# Teste la route /ask avec un champ question manquant.
def test_ask_missing_question() -> None:
    # Affiche le nom du test.
    print_separator("TEST : POST /ask (champ manquant)")

    # Envoie un JSON qui ne contient pas le champ question attendu.
    response = requests.post(
        # URL complete de la route.
        f"{BASE_URL}/ask",
        # Corps JSON incorrect.
        json={"wrong_field": "test"},
        # Timeout court car la validation doit echouer rapidement.
        timeout=10,
    )

    # Affiche le code HTTP retourne.
    print(f"Status code : {response.status_code}")
    # Affiche le texte brut de la reponse.
    print(response.text)


# Teste la route /ask avec un payload vide.
def test_ask_empty_payload() -> None:
    # Affiche le nom du test.
    print_separator("TEST : POST /ask (payload vide)")

    # Envoie un JSON vide.
    response = requests.post(
        # URL complete de la route.
        f"{BASE_URL}/ask",
        # Corps JSON vide.
        json={},
        # Timeout court car la validation doit echouer rapidement.
        timeout=10,
    )

    # Affiche le code HTTP retourne.
    print(f"Status code : {response.status_code}")
    # Affiche le texte brut de la reponse.
    print(response.text)


# Teste la route /ask avec un JSON invalide.
def test_ask_invalid_json() -> None:
    # Affiche le nom du test.
    print_separator("TEST : POST /ask (JSON invalide)")

    # Envoie volontairement un JSON mal forme.
    response = requests.post(
        # URL complete de la route.
        f"{BASE_URL}/ask",
        # Corps texte invalide, il manque une accolade fermante.
        data='{"question":"Quels événements à Toulouse ?"',

        # Precise a l'API que le contenu est cense etre du JSON.
        headers={
            # Type de contenu envoye.
            "Content-Type": "application/json",
        },
        # Timeout court car la validation doit echouer rapidement.
        timeout=10,
    )

    # Affiche le code HTTP retourne.
    print(f"Status code : {response.status_code}")
    # Affiche le texte brut de la reponse.
    print(response.text)


# Teste la route /rebuild sans cle API.
def test_rebuild_without_key() -> None:
    # Affiche le nom du test.
    print_separator("TEST : POST /rebuild (sans cle API)")

    # Envoie une requete POST sans header X-API-Key.
    response = requests.post(
        # URL complete de la route.
        f"{BASE_URL}/rebuild",
        # Timeout court car la requete doit etre refusee si une cle est configuree.
        timeout=10,
    )

    # Affiche le code HTTP retourne.
    print(f"Status code : {response.status_code}")
    # Affiche le texte brut de la reponse.
    print(response.text)


# Lance tous les tests manuels les uns apres les autres.
def main() -> None:
    # Teste la route de sante.
    test_health()

    # Teste une question valide.
    test_ask_valid_question()

    # Teste une question vide.
    test_ask_empty_question()

    # Teste une question avec seulement des espaces.
    test_ask_spaces_only_question()

    # Teste l'absence du champ question.
    test_ask_missing_question()

    # Teste un corps JSON vide.
    test_ask_empty_payload()

    # Teste un JSON invalide.
    test_ask_invalid_json()

    # Teste la reconstruction sans cle API.
    test_rebuild_without_key()


# Verifie que le script est lance directement.
if __name__ == "__main__":
    # Lance la fonction principale.
    main()