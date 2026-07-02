# Tester la recherche semantique

# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe Path pour manipuler le chemin de l'index FAISS.
from pathlib import Path

# Importe FAISS pour charger l'index vectoriel local.
from langchain_community.vectorstores import FAISS

# Importe la classe qui cree les embeddings avec Mistral.
from src.mistral_embeddings import MistralEmbeddings

# Definit le chemin ou l'index FAISS est sauvegarde.
VECTORSTORE_PATH = Path("vectorstore/faiss_index")


# Fonction principale du script de test.
def main() -> None:
    # Initialise les embeddings Mistral avec le modele mistral-embed.
    embeddings = MistralEmbeddings(model="mistral-embed")

    # Charge l'index FAISS depuis le disque.
    vectorstore = FAISS.load_local(
        # Convertit le chemin Path en texte.
        str(VECTORSTORE_PATH),
        # Fournit le meme modele d'embeddings que celui utilise pour l'index.
        embeddings,
        # Autorise le chargement local de l'index FAISS par LangChain.
        allow_dangerous_deserialization=True,
    )

    # Definit une question de test pour la recherche semantique.
    query = "Quels événements autour du tourisme vont avoir lieu à Toulouse ?"

    # Cherche les 5 resultats les plus proches et recupere aussi leur score.
    results = vectorstore.similarity_search_with_score(query, k=5)

    # Affiche la question testee.
    print(f"Question : {query}")
    # Affiche une ligne vide pour rendre la sortie plus lisible.
    print()

    # Parcourt les resultats avec un numero qui commence a 1.
    for index, (doc, score) in enumerate(results, start=1):
        # Affiche un separateur visuel.
        print("=" * 80)
        # Affiche le numero du resultat.
        print(f"Resultat {index}")
        # Affiche le score FAISS associe au resultat.
        print(f"Score FAISS : {score}")
        # Affiche le titre stocke dans les metadonnees.
        print(f"Titre : {doc.metadata.get('title')}")
        # Affiche la ville stockee dans les metadonnees.
        print(f"Ville : {doc.metadata.get('city')}")
        # Affiche la date stockee dans les metadonnees.
        print(f"Date : {doc.metadata.get('begin')}")
        # Affiche le lieu stocke dans les metadonnees.
        print(f"Lieu : {doc.metadata.get('venue')}")
        # Affiche l'URL source stockee dans les metadonnees.
        print(f"URL : {doc.metadata.get('url')}")
        # Affiche une ligne vide.
        print()
        # Affiche les 700 premiers caracteres du contenu du document.
        print(doc.page_content[:700])


# Verifie que le script est lance directement.
if __name__ == "__main__":
    # Lance la fonction principale.
    main()