# Importe json pour lire le fichier d'embeddings au format JSON.
import json
# Importe Path pour manipuler proprement un chemin de fichier.
from pathlib import Path


# Definit le chemin du fichier contenant les embeddings generes.
EMBEDDINGS_PATH = Path("data/processed/events_embeddings.json")


# Teste que le fichier d'embeddings existe bien.
def test_embeddings_file_exists():
    # Verifie la presence du fichier sur le disque.
    assert EMBEDDINGS_PATH.exists()


# Teste que le fichier d'embeddings contient la structure attendue.
def test_embeddings_have_expected_structure():
    # Lit le fichier JSON et le transforme en objet Python.
    data = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))

    # Verifie que le fichier contient au moins un element.
    assert len(data) > 0

    # Recupere le premier element pour verifier sa structure.
    first = data[0]

    # Verifie que l'identifiant de l'evenement est present.
    assert "uid" in first
    # Verifie que le titre de l'evenement est present.
    assert "title" in first
    # Verifie que le texte utilise pour l'embedding est present.
    assert "text_for_embedding" in first
    # Verifie que le vecteur d'embedding est present.
    assert "embedding" in first
    # Verifie que le nom du modele d'embedding est present.
    assert "embedding_model" in first
    # Verifie que la dimension du vecteur est presente.
    assert "embedding_dim" in first

    # Verifie que le modele utilise est bien mistral-embed.
    assert first["embedding_model"] == "mistral-embed"
    # Verifie que l'embedding est une liste de nombres.
    assert isinstance(first["embedding"], list)
    # Verifie que la longueur du vecteur correspond a la dimension annoncee.
    assert len(first["embedding"]) == first["embedding_dim"]
    # Verifie que la dimension attendue pour mistral-embed est 1024.
    assert first["embedding_dim"] == 1024