import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Vérification de l'environnement...")

    import pandas as pd
    import requests
    import faiss
    import mistralai

    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_mistralai import ChatMistralAI


    print("pandas:", pd.__version__)
    print("requests:", requests.__version__)
    print("faiss:", faiss.__version__)

    mistral_key = os.getenv("MISTRAL_API_KEY")
    if mistral_key:
        print("MISTRAL_API_KEY: OK")
    else:
        print("MISTRAL_API_KEY: absente, normal si tu n'as pas encore configuré la clé")

    print("Imports LangChain / Faiss / Mistral: OK")

if __name__ == "__main__":
    main()