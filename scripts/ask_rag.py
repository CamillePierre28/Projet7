# Ceci est script de test manuel. Il sert a poser une question au systeme RAG sans passer encore par une API. Il verifie que :
# - FAISS se charge correctement
# - la question est vectorisee
# - Mistral genere une reponse naturelle
# - les metadonnees des sources sont retournees

# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe la classe principale du systeme RAG.
from src.rag import EventRAG


# Fonction principale du script.
def main() -> None:
    # Initialise le systeme RAG complet.
    rag = EventRAG()

    # Definit une question de test.
    question = "Quels événements culturels sont prévus à Toulouse ?"

    # Envoie la question au RAG et recupere le resultat.
    result = rag.answer(question)

    # Affiche le titre de la section question.
    print("Question :")
    # Affiche la question posee.
    print(result["question"])
    # Affiche une ligne vide pour la lisibilite.
    print()

    # Affiche le titre de la section reponse.
    print("Reponse :")
    # Affiche la reponse generee par le modele.
    print(result["answer"])
    # Affiche une ligne vide pour la lisibilite.
    print()

    # Affiche le titre de la section sources.
    print("Sources :")
    # Parcourt toutes les sources utilisees par le RAG.
    for source in result["sources"]:
        # Affiche chaque source sous forme de dictionnaire.
        print("-", source)


# Verifie que le script est lance directement.
if __name__ == "__main__":
    # Lance la fonction principale.
    main()