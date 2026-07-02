# Importe pandas pour creer un faux tableau d'evenements.
import pandas as pd

# Importe les fonctions de transformation et de decoupage des documents.
from src.chunking import dataframe_to_documents, split_documents


# Teste que les lignes d'un DataFrame deviennent des documents LangChain.
def test_dataframe_to_documents_creates_documents():
    # Cree un DataFrame avec un evenement de test.
    df = pd.DataFrame(
        [
            {
                # Identifiant unique de l'evenement.
                "uid": "1",
                # Titre de l'evenement.
                "title": "Concert test",
                # Date de debut.
                "begin": "2025-01-01",
                # Lieu de l'evenement.
                "venue": "Salle test",
                # Ville de l'evenement.
                "city": "Toulouse",
                # URL source.
                "url": "https://example.com",
                # Texte qui sera place dans le contenu du document.
                "text_for_embedding": "Titre : Concert test\nDescription : Un tres beau concert a Toulouse.",
            }
        ]
    )

    # Transforme le DataFrame en documents LangChain.
    docs = dataframe_to_documents(df)

    # Verifie qu'un seul document est cree.
    assert len(docs) == 1
    # Verifie que l'UID est bien conserve dans les metadonnees.
    assert docs[0].metadata["uid"] == "1"
    # Verifie que la ville est bien conservee dans les metadonnees.
    assert docs[0].metadata["city"] == "Toulouse"
    # Verifie que le titre apparait bien dans le contenu du document.
    assert "Concert test" in docs[0].page_content


# Teste que les documents longs sont bien decoupes en plusieurs chunks.
def test_split_documents_creates_chunks():
    # Cree un DataFrame avec un texte volontairement long.
    df = pd.DataFrame(
        [
            {
                # Identifiant unique de l'evenement.
                "uid": "1",
                # Titre de l'evenement.
                "title": "Festival test",
                # Date de debut.
                "begin": "2025-01-01",
                # Lieu de l'evenement.
                "venue": "Lieu test",
                # Ville de l'evenement.
                "city": "Toulouse",
                # URL source.
                "url": "https://example.com",
                # Texte repete pour forcer le decoupage en plusieurs morceaux.
                "text_for_embedding": "Texte tres long. " * 200,
            }
        ]
    )

    # Transforme la ligne en document LangChain.
    docs = dataframe_to_documents(df)
    # Decoupe le document en chunks plus petits.
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=30)

    # Verifie que le texte long a bien produit plusieurs chunks.
    assert len(chunks) > 1
    # Verifie que chaque chunk conserve l'UID de l'evenement d'origine.
    assert all(chunk.metadata["uid"] == "1" for chunk in chunks)