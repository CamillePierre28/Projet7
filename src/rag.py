# Ici c'est le coeur du chatbot. Il fait 4 choses : 
# 1. charge l’index FAISS local
# 2. utilise les embeddings Mistral pour rechercher les documents proches
# 3. envoie les documents récupérés à un LLM Mistral
# 4. retourne une réponse + les sources utilisées
# Techniquement il a été mis en place une chaîne LangChain : question utilisateur -> recherche FAISS -> sélection des chunks pertinents -> construction d'un contexte -> prompt RAG -> Mistral Chat -> réponse finale
# Un filtre métier a été ajouté "is_future_question()", si la question contient "prévu, à venir, prochain, demain, week-end..." alors le système filtre les événements passés et garde seulement les événements futurs. 


from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser

from src.mistral_embeddings import MistralEmbeddings

# Charge les variables d’environnement depuis le fichier .env.
load_dotenv()

# Chemin vers l’index FAISS sauvegardé localement.
VECTORSTORE_PATH = Path("vectorstore/faiss_index")

# Fonction qui détecte si la question utilisateur concerne des événements futurs.
def is_future_question(question: str) -> bool:
    # Liste de mots-clés indiquant que l’utilisateur cherche des événements à venir.
    keywords = ["prévu", "prévus", "à venir", "prochain", "prochaine", "bientôt", "demain", "week-end"]
    # Convertit la question en minuscules pour rendre la recherche insensible à la casse.
    question_lower = question.lower()
    # Retourne True si au moins un mot-clé est présent dans la question.
    return any(keyword in question_lower for keyword in keywords)


# Fonction qui filtre une liste de documents pour garder uniquement les événements futurs.
def filter_future_documents(documents: list[Document]) -> list[Document]:
    # Récupère la date et l’heure actuelles en UTC.
    now = datetime.now(timezone.utc)
    # Initialise une liste vide pour stocker les documents conservés.
    filtered = []

    # Parcourt chaque document récupéré par FAISS.
    for doc in documents:
        # Récupère la date de début de l’événement depuis les métadonnées.
        begin = doc.metadata.get("begin")

        # Essaie de convertir la date de début en objet datetime Python.
        try:
            # Convertit la date ISO en datetime. replace("Z", "+00:00") permet de gérer les dates terminées par Z, qui signifie UTC.
            event_date = datetime.fromisoformat(str(begin).replace("Z", "+00:00"))
        # Si la date est invalide, on ignore ce document.
        except ValueError:
            continue

        # Si la date de l’événement est supérieure ou égale à maintenant, alors l’événement est à venir.
        if event_date >= now:
            # Ajoute le document à la liste des documents filtrés.
            filtered.append(doc)

    # Retourne uniquement les documents correspondant à des événements futurs.
    return filtered

# Classe principale du système RAG dédié aux événements.
class EventRAG:
    # Méthode appelée automatiquement quand on crée un objet EventRAG.
    def __init__(self) -> None:
        # Initialise le modèle d’embedding Mistral personnalisé. Il sera utilisé pour transformer la question utilisateur en vecteur.
        self.embeddings = MistralEmbeddings()

        # Charge l’index FAISS sauvegardé localement.
        self.vectorstore = FAISS.load_local(
            # Convertit le chemin Path en chaîne de caractères.
            str(VECTORSTORE_PATH),
            # Fournit le modèle d’embedding utilisé pour comparer la question aux documents.
            self.embeddings,
            # Autorise la désérialisation de l’index FAISS. C’est nécessaire avec LangChain pour recharger un index local.
            allow_dangerous_deserialization=True,
        )

        # Initialise le modèle de chat Mistral utilisé pour générer la réponse finale.
        self.llm = ChatMistralAI(
            # Récupère le modèle depuis le .env. Si aucune valeur n’est définie, utilise mistral-small-latest.
            model=os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest"),
            # Température faible pour obtenir des réponses plus stables et limiter les inventions.
            temperature=0.2,
        )

        # Crée le prompt utilisé par le chatbot.
        self.prompt = ChatPromptTemplate.from_template(
            """
Tu es un assistant spécialisé dans la recommandation d'événements culturels en Occitanie.

Réponds uniquement à partir du contexte fourni.
Si le contexte ne contient pas l'information, dis que tu ne peux pas répondre précisément.

Question utilisateur :
{question}

Contexte :
{context}

Réponse attendue :
- réponse claire et concise
- cite les événements utiles avec titre, ville, date et lieu si disponibles
- ne fabrique jamais d'événement
"""
        )

        # Crée la chaîne LangChain complète : prompt -> modèle Mistral -> extraction du texte final.
        self.chain = self.prompt | self.llm | StrOutputParser()

    # Méthode qui récupère les documents les plus pertinents pour une question.
    def retrieve(self, question: str) -> list[Document]:
        # Nombre de documents à récupérer au départ depuis FAISS. Valeur configurable dans le fichier .env.
        fetch_k = int(os.getenv("RAG_FETCH_K", "20"))
        # Nombre final de documents à garder pour le contexte. Valeur configurable dans le fichier .env.
        top_k = int(os.getenv("RAG_TOP_K", "5"))

        # Recherche les documents les plus proches sémantiquement de la question.
        docs = self.vectorstore.similarity_search(question, k=fetch_k)

        # Si la question semble concerner des événements futurs, on filtre les résultats pour supprimer les événements passés.
        if is_future_question(question):
            # Garde uniquement les documents dont la date est future.
            docs = filter_future_documents(docs)

        # Retourne seulement les top_k premiers documents.
        return docs[:top_k]

    # Méthode qui génère une réponse complète à partir d’une question utilisateur.
    def answer(self, question: str) -> dict:
        # Récupère les documents pertinents depuis FAISS.
        docs = self.retrieve(question)

        # Construit le contexte textuel envoyé au LLM.
        context = "\n\n".join(
            [
                # Ajoute le titre de l’événement.
                f"Titre : {doc.metadata.get('title')}\n"
                # Ajoute la ville de l’événement.
                f"Ville : {doc.metadata.get('city')}\n"
                # Ajoute la date de début de l’événement.
                f"Date : {doc.metadata.get('begin')}\n"
                # Ajoute le lieu de l’événement.
                f"Lieu : {doc.metadata.get('venue')}\n"
                # Ajoute l’URL source de l’événement.
                f"URL : {doc.metadata.get('url')}\n"
                # Ajoute le contenu textuel du document.
                f"Contenu : {doc.page_content}"
                # Répète cette structure pour chaque document récupéré.
                for doc in docs
            ]
        )

        # Envoie la question et le contexte dans la chaîne LangChain.
        response = self.chain.invoke(
            {
                # Question posée par l’utilisateur.
                "question": question,
                # Documents récupérés et formatés en contexte.
                "context": context,
            }
        )

        # Retourne un dictionnaire contenant la question, la réponse et les sources utilisées.
        return {
            # Question originale de l’utilisateur.
            "question": question,
            # Réponse générée par le modèle Mistral.
            "answer": response,
            # Liste des sources utilisées pour construire la réponse.
            "sources": [
                {
                    # Identifiant unique de l’événement.
                    "uid": doc.metadata.get("uid"),
                    # Titre de l’événement.
                    "title": doc.metadata.get("title"),
                    # Ville de l’événement.
                    "city": doc.metadata.get("city"),
                    # Date de début.
                    "begin": doc.metadata.get("begin"),
                    # Lieu.
                    "venue": doc.metadata.get("venue"),
                    # URL de l’événement.
                    "url": doc.metadata.get("url"),
                }
                # Crée une source pour chaque document récupéré.
                for doc in docs
            ],
        }
    

