# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe os pour lire les variables d'environnement.
import os
# Importe subprocess pour lancer des scripts Python depuis l'API.
import subprocess
# Importe sys pour recuperer le chemin de l'interpreteur Python courant.
import sys

# Importe load_dotenv pour charger les variables du fichier .env.
from dotenv import load_dotenv
# Importe les objets FastAPI necessaires pour creer l'API et gerer les erreurs.
from fastapi import FastAPI, Header, HTTPException, status
# Importe BaseModel et Field pour definir les formats d'entree et de sortie.
from pydantic import BaseModel, Field

# Importe la fonction d'evaluation automatique des reponses.
from src.evaluation import evaluate_answer_online
# Importe la classe principale du systeme RAG.
from src.rag import EventRAG

# Charge les variables d'environnement depuis le fichier .env.
load_dotenv()

# Recupere la cle API qui protege la route /rebuild.
REBUILD_API_KEY = os.getenv("REBUILD_API_KEY")

# Cree l'application FastAPI.
app = FastAPI(
    # Titre affiche dans la documentation interactive.
    title="Puls-Events RAG API",
    # Description affichee dans la documentation interactive.
    description=(
        "API REST locale permettant d'interroger un chatbot RAG "
        "sur des événements culturels en Occitanie."
    ),
    # Version de l'API.
    version="1.0.0",
)

# Variable globale qui contiendra le systeme RAG charge au demarrage.
rag: EventRAG | None = None


# Definit le format attendu pour une requete /ask.
class AskRequest(BaseModel):
    # Champ question obligatoire, avec une longueur minimale de 1 caractere.
    question: str = Field(
        # Les trois points indiquent que le champ est obligatoire.
        ...,
        # Refuse les chaines totalement vides.
        min_length=1,
        # Description visible dans la documentation FastAPI.
        description="Question utilisateur à poser au chatbot RAG.",
        # Exemple visible dans la documentation FastAPI.
        examples=["Quels événements culturels sont prévus à Toulouse ?"],
    )


# Definit le format d'une source retournee par le RAG.
class Source(BaseModel):
    # Identifiant unique de l'evenement.
    uid: str | None = None
    # Titre de l'evenement.
    title: str | None = None
    # Ville de l'evenement.
    city: str | None = None
    # Date de debut de l'evenement.
    begin: str | None = None
    # Lieu de l'evenement.
    venue: str | None = None
    # URL source de l'evenement.
    url: str | None = None


# Definit le format de l'evaluation automatique renvoyee par l'API.
class EvaluationResponse(BaseModel):
    # Nombre de sources utilisees dans la reponse.
    sources_count: int
    # Ville attendue detectee dans la question.
    expected_city: str
    # Indique si la ville attendue est bien presente.
    city_ok: bool
    # Mot-cle attendu detecte dans la question.
    expected_keyword: str
    # Indique si le mot-cle attendu est bien present.
    keyword_ok: bool
    # Indique si la question demande des evenements futurs.
    requires_future: bool
    # Indique si les sources respectent la contrainte de futur.
    future_ok: bool
    # Classe finale de l'evaluation.
    classification: str


# Definit le format complet de la reponse de /ask.
class AskResponse(BaseModel):
    # Question posee par l'utilisateur.
    question: str
    # Reponse generee par le RAG.
    answer: str
    # Liste des sources utilisees.
    sources: list[Source]
    # Evaluation automatique de la reponse.
    evaluation: EvaluationResponse


# Definit le format de la reponse de /health.
class HealthResponse(BaseModel):
    # Statut general de l'API.
    status: str
    # Indique si le systeme RAG est charge.
    rag_loaded: bool


# Definit le format de la reponse de /rebuild.
class RebuildResponse(BaseModel):
    # Statut de l'operation.
    status: str
    # Message explicatif.
    message: str


# Execute cette fonction au demarrage de l'application FastAPI.
@app.on_event("startup")
def load_rag() -> None:
    # Indique que l'on veut modifier la variable globale rag.
    global rag

    # Essaie de charger le systeme RAG.
    try:
        # Instancie le RAG, ce qui charge notamment l'index FAISS.
        rag = EventRAG()
        # Affiche un message de succes dans les logs.
        print("RAG charge avec succes")

    # Capture les erreurs pour eviter que l'API plante totalement au demarrage.
    except Exception as exc:
        # Met rag a None pour signaler qu'il n'est pas disponible.
        rag = None
        # Affiche l'erreur dans les logs.
        print(f"Erreur au chargement du RAG : {exc}")


# Declare la route GET /health.
@app.get(
    # Chemin de la route.
    "/health",
    # Modele de reponse attendu.
    response_model=HealthResponse,
    # Resume affiche dans la documentation FastAPI.
    summary="Vérifier l'état de l'API",
)
def health() -> HealthResponse:
    # Retourne l'etat de l'API.
    return HealthResponse(
        # L'API repond, donc le statut general est ok.
        status="ok",
        # True si le RAG est charge, False sinon.
        rag_loaded=rag is not None,
    )


# Declare la route POST /ask.
@app.post(
    # Chemin de la route.
    "/ask",
    # Modele de reponse attendu.
    response_model=AskResponse,
    # Resume affiche dans la documentation FastAPI.
    summary="Poser une question au chatbot RAG",
    # Description detaillee affichee dans la documentation FastAPI.
    description=(
        "Prend une question utilisateur, recherche les événements pertinents "
        "dans FAISS, génère une réponse avec Mistral, puis retourne aussi "
        "une évaluation automatique légère de la qualité de la réponse."
    ),
)
def ask(request: AskRequest) -> AskResponse:
    # Si le RAG n'est pas charge, l'API ne peut pas repondre aux questions.
    if rag is None:
        # Retourne une erreur HTTP 503 : service indisponible.
        raise HTTPException(
            # Code HTTP indiquant que le service n'est pas pret.
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            # Message d'erreur envoye au client.
            detail="Le système RAG n'est pas chargé.",
        )

    # Supprime les espaces inutiles autour de la question.
    question = request.question.strip()

    # Si la question devient vide apres suppression des espaces, on la refuse.
    if not question:
        # Retourne une erreur HTTP 400 : requete invalide.
        raise HTTPException(
            # Code HTTP indiquant une mauvaise requete.
            status_code=status.HTTP_400_BAD_REQUEST,
            # Message d'erreur envoye au client.
            detail="La question ne peut pas être vide.",
        )

    # Envoie la question au systeme RAG.
    result = rag.answer(question)

    # Evalue automatiquement la reponse et les sources.
    evaluation = evaluate_answer_online(
        # Question utilisee par le RAG.
        question=result["question"],
        # Reponse generee.
        answer=result["answer"],
        # Sources retrouvees.
        sources=result["sources"],
    )

    # Construit la reponse finale conforme au modele AskResponse.
    return AskResponse(
        # Question originale nettoyee.
        question=result["question"],
        # Reponse generee par Mistral via le RAG.
        answer=result["answer"],
        # Sources utilisees pour construire la reponse.
        sources=result["sources"],
        # Evaluation automatique.
        evaluation=evaluation,
    )


# Declare la route POST /rebuild.
@app.post(
    # Chemin de la route.
    "/rebuild",
    # Modele de reponse attendu.
    response_model=RebuildResponse,
    # Resume affiche dans la documentation FastAPI.
    summary="Reconstruire la base vectorielle",
    # Description detaillee affichee dans la documentation FastAPI.
    description=(
        "Endpoint sensible permettant de relancer la collecte, "
        "le preprocessing et la reconstruction de l'index FAISS. "
        "Protégé par une clé API via le header X-API-Key."
    ),
)
def rebuild(
    # Lit le header X-API-Key envoye par le client.
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> RebuildResponse:
    # Si une cle est configuree, elle doit correspondre a celle envoyee par le client.
    if REBUILD_API_KEY and x_api_key != REBUILD_API_KEY:
        # Retourne une erreur HTTP 401 si la cle est absente ou incorrecte.
        raise HTTPException(
            # Code HTTP indiquant que l'utilisateur n'est pas autorise.
            status_code=status.HTTP_401_UNAUTHORIZED,
            # Message d'erreur envoye au client.
            detail="Clé API invalide pour /rebuild.",
        )

    # Essaie de relancer toute la chaine de reconstruction.
    try:
        # Lance le script qui recupere et nettoie les evenements.
        subprocess.run(
            # Utilise le meme interpreteur Python que l'application courante.
            [sys.executable, "-m", "scripts.fetch_openagenda"],
            # Leve une erreur si le script echoue.
            check=True,
        )

        # Lance le script qui reconstruit l'index FAISS.
        subprocess.run(
            # Utilise le meme interpreteur Python que l'application courante.
            [sys.executable, "-m", "scripts.build_vectorstore"],
            # Leve une erreur si le script echoue.
            check=True,
        )

        # Indique que l'on veut remplacer la variable globale rag.
        global rag
        # Recharge le RAG pour utiliser le nouvel index FAISS.
        rag = EventRAG()

        # Retourne une reponse de succes.
        return RebuildResponse(
            # Statut de l'operation.
            status="success",
            # Message explicatif.
            message="Index FAISS reconstruit et RAG rechargé.",
        )

    # Capture les erreurs venant des scripts lances avec subprocess.
    except subprocess.CalledProcessError as exc:
        # Retourne une erreur HTTP 500 si la reconstruction echoue.
        raise HTTPException(
            # Code HTTP indiquant une erreur interne.
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            # Message d'erreur avec le detail de l'echec.
            detail=f"Erreur pendant la reconstruction : {exc}",
        )