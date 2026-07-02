# Construction de l'index FAISS

# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe Path pour manipuler les chemins de fichiers.
from pathlib import Path

# Importe pandas pour lire le fichier CSV des evenements nettoyes.
import pandas as pd
# Importe FAISS pour creer l'index vectoriel.
from langchain_community.vectorstores import FAISS

# Importe les fonctions qui transforment le DataFrame en documents puis en chunks.
from src.chunking import dataframe_to_documents, split_documents
# Importe la classe qui cree les embeddings avec Mistral.
from src.mistral_embeddings import MistralEmbeddings

# Chemin du fichier CSV contenant les evenements nettoyes.
PROCESSED_PATH = Path("data/processed/events_processed.csv")
# Chemin ou l'index FAISS sera sauvegarde.
VECTORSTORE_PATH = Path("vectorstore/faiss_index")


# Fonction principale qui construit l'index FAISS.
def main() -> None:
    # Affiche le debut du chargement des donnees.
    print("Chargement des événements nettoyés...")
    # Lit le CSV des evenements nettoyes dans un DataFrame pandas.
    df = pd.read_csv(PROCESSED_PATH)

    # Affiche le nombre d'événements chargés.
    print(f"{len(df)} événements chargés")

    # Transforme chaque ligne du DataFrame en document LangChain.
    documents = dataframe_to_documents(df)
    # Decoupe les documents en chunks plus petits.
    chunks = split_documents(documents)

    # Affiche le nombre de documents crees.
    print(f"{len(documents)} documents événement créés")
    # Affiche le nombre de chunks créés.
    print(f"{len(chunks)} chunks créés")

    # Recupere les identifiants uniques presents dans les chunks.
    unique_event_ids = {chunk.metadata["uid"] for chunk in chunks}
    # Affiche combien d'evenements differents sont representes dans les chunks.
    print(f"{len(unique_event_ids)} événements représentés dans les chunks")

    # Verifie que chaque evenement du CSV a bien ete represente dans au moins un chunk.
    if len(unique_event_ids) != len(df):
        # Arrete le script si certains evenements ont disparu pendant le chunking.
        raise RuntimeError(
            f"Tous les événements ne sont pas indexés : "
            f"{len(unique_event_ids)} / {len(df)}"
        )

    # Initialise le modele d'embeddings Mistral.
    embeddings = MistralEmbeddings(
        # Nom du modele d'embeddings.
        model="mistral-embed",
        # Nombre de textes envoyes par lot a Mistral.
        batch_size=32,
    )

    # Affiche le debut de la construction de l'index.
    print("Construction de l'index FAISS avec embeddings Mistral...")
    # Cree l'index FAISS a partir des chunks et des embeddings.
    vectorstore = FAISS.from_documents(
        # Documents decoupes a indexer.
        documents=chunks,
        # Modele utilise pour transformer les chunks en vecteurs.
        embedding=embeddings,
    )

    # Cree le dossier de sortie si necessaire.
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    # Sauvegarde l'index FAISS sur le disque.
    vectorstore.save_local(str(VECTORSTORE_PATH))

    # Affiche le chemin ou l'index a ete sauvegarde.
    print(f"Index FAISS sauvegardé dans : {VECTORSTORE_PATH}")
    # Affiche un message de succes.
    print("Indexation terminée avec succès")


# Verifie que le script est lance directement.
if __name__ == "__main__":
    # Lance la fonction principale.
    main()