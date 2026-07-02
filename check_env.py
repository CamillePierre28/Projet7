# Importe le module os pour lire les variables d'environnement.
import os

# Importe load_dotenv pour charger les variables du fichier .env.
from dotenv import load_dotenv

# Charge les variables definies dans le fichier .env dans l'environnement Python.
load_dotenv()


# Definit la fonction principale du script.
def main():
    # Affiche un message pour indiquer que la verification commence.
    print("Vérification de l'environnement...")

    # Importe pandas pour verifier que la librairie de manipulation de donnees est installee.
    import pandas as pd
    # Importe requests pour verifier que la librairie d'appels API est installee.
    import requests
    # Importe faiss pour verifier que le moteur vectoriel est installe.
    import faiss
    # Importe mistralai pour verifier que le SDK Mistral est installe.
    import mistralai

    # Importe FAISS version LangChain pour verifier l'integration vectorstore.
    from langchain_community.vectorstores import FAISS
    # Importe HuggingFaceEmbeddings pour verifier une integration embeddings possible avec LangChain.
    from langchain_huggingface import HuggingFaceEmbeddings
    # Importe ChatMistralAI pour verifier que LangChain peut utiliser Mistral en chat.
    from langchain_mistralai import ChatMistralAI

    # Affiche la version de pandas installee.
    print("pandas:", pd.__version__)
    # Affiche la version de requests installee.
    print("requests:", requests.__version__)
    # Affiche la version de faiss installee.
    print("faiss:", faiss.__version__)

    # Recupere la cle API Mistral depuis les variables d'environnement.
    mistral_key = os.getenv("MISTRAL_API_KEY")
    # Si la cle existe, on confirme qu'elle est presente.
    if mistral_key:
        # Affiche que la cle Mistral est bien configuree.
        print("MISTRAL_API_KEY: OK")
    # Sinon, on signale qu'elle est absente.
    else:
        # Affiche que la cle Mistral manque dans l'environnement.
        print("MISTRAL_API_KEY: absente")

    # Affiche un message final si tous les imports precedents ont fonctionne.
    print("Imports LangChain / Faiss / Mistral: OK")


# Verifie que le fichier est lance directement, et pas seulement importe.
if __name__ == "__main__":
    # Lance la fonction principale.
    main()