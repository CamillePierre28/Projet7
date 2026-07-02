# Chunking des evenements

# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe pandas pour lire les evenements sous forme de DataFrame.
import pandas as pd
# Importe Document, le format standard utilise par LangChain.
from langchain_core.documents import Document
# Importe un decoupeur de texte intelligent fourni par LangChain.
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Transforme un DataFrame d'evenements en documents LangChain.
def dataframe_to_documents(df: pd.DataFrame) -> list[Document]:
    # Cree une liste vide qui contiendra les documents.
    documents = []

    # Parcourt chaque ligne du DataFrame.
    for _, row in df.iterrows():
        # Prepare les metadonnees de l'evenement.
        metadata = {
            # Identifiant unique de l'evenement, force en texte.
            "uid": str(row["uid"]),
            # Titre de l'evenement.
            "title": row.get("title", ""),
            # Date de debut, forcee en texte.
            "begin": str(row.get("begin", "")),
            # Date de fin, forcee en texte.
            "end": str(row.get("end", "")),
            # Nom du lieu.
            "venue": row.get("venue", ""),
            # Adresse du lieu.
            "address": row.get("address", ""),
            # Ville de l'evenement.
            "city": row.get("city", ""),
            # Departement de l'evenement.
            "department": row.get("department", ""),
            # Region de l'evenement.
            "region": row.get("region", ""),
            # URL de la source.
            "url": row.get("url", ""),
        }

        # Ajoute un document LangChain dans la liste.
        documents.append(
            # Cree le document avec un contenu principal et des metadonnees.
            Document(
                # Texte utilise pour la recherche semantique.
                page_content=row["text_for_embedding"],
                # Informations utiles pour afficher les sources.
                metadata=metadata,
            )
        )

    # Retourne la liste complete des documents.
    return documents


# Decoupe les documents en morceaux plus petits appeles chunks.
def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    # Configure le decoupeur de texte.
    splitter = RecursiveCharacterTextSplitter(
        # Taille maximale d'un chunk en caracteres.
        chunk_size=chunk_size,
        # Nombre de caracteres repris entre deux chunks voisins.
        chunk_overlap=chunk_overlap,
        # Ordre des separateurs essayes pour couper proprement le texte.
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Decoupe tous les documents et retourne la liste des chunks.
    return splitter.split_documents(documents)