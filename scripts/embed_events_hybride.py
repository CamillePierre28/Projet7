# Conversion des descriptions des événements en vecteurs numériques grâce au modèle d'embedding de Mistral AI. Approche hybride pour l'intégration des métadonnées. 

from __future__ import annotations
import json
import os
from pathlib import Path
from time import sleep
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral

# Charge les variables d'environnement du fichier .env.
load_dotenv()


# Chemin du fichier contenant les événements nettoyés.
PROCESSED_PATH = Path("data/processed/events_processed.csv")
# Chemin du fichier où seront sauvegardés les embeddings.
EMBEDDINGS_PATH = Path("data/processed/events_embeddings.json")


# Nom du modèle d'embedding utilisé.
MODEL_NAME = "mistral-embed"
# Nom de la stratégie d'embedding utilisée. Cela permet de savoir plus tard comment les vecteurs ont été créés.
EMBEDDING_STRATEGY = "hybrid_metadata_v1"

# Nombre d'événements envoyés à l'API à chaque requête.
BATCH_SIZE = int(os.getenv("MISTRAL_EMBED_BATCH_SIZE", "8"))
# Temps d'attente entre deux appels API.
SLEEP_SECONDS = float(os.getenv("MISTRAL_EMBED_SLEEP_SECONDS", "2"))
# Nombre maximum de tentatives en cas de rate limit.
MAX_RETRIES = 6


# Fonction utilitaire qui transforme une valeur vide en chaîne vide afin d'éviter les erreurs.
def safe_value(value) -> str:

    # Si la valeur est NaN (valeur manquante pandas) on retourne une chaîne vide.
    if pd.isna(value):
        return ""
    # Sinon on convertit la valeur en texte puis on supprime les espaces inutiles.
    return str(value).strip()


# Fonction qui construit le texte enrichi utilisé pour générer l'embedding.
def build_hybrid_embedding_text(event: dict) -> str:
    # Récupération des différentes métadonnées.
    title = safe_value(event.get("title"))
    begin = safe_value(event.get("begin"))
    venue = safe_value(event.get("venue"))
    city = safe_value(event.get("city"))
    # Récupération du texte principal.
    text = safe_value(event.get("text_for_embedding"))

    # Construction du document enrichi. Ce texte sera envoyé à Mistral.
    return f"""
Titre de l'événement : {title}

Date de début : {begin}

Lieu : {venue}
Ville : {city}

Description de l'événement :
{text}
""".strip()


# Fonction qui charge les embeddings déjà calculés.
def load_existing_embeddings() -> dict[str, dict]:
    # Si aucun fichier n'existe encore, on retourne un dictionnaire vide.
    if not EMBEDDINGS_PATH.exists():
        return {}

    # Lecture du fichier JSON.
    data = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))

    # Création d'un dictionnaire indexé par UID. On conserve uniquement les embeddings utilisant la stratégie actuelle.
    return {
        str(item["uid"]): item
        for item in data
        if item.get("embedding_strategy")
        == EMBEDDING_STRATEGY
    }


# Fonction de sauvegarde des embeddings.
def save_embeddings(items: list[dict]) -> None:
    # Création automatique du dossier parent.
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Sauvegarde du JSON.
    EMBEDDINGS_PATH.write_text(
        # Conversion des objets Python en JSON.
        json.dumps(items, ensure_ascii=False, indent=2),
        # Encodage UTF-8.
        encoding="utf-8",
    )


# Fonction qui envoie un batch à Mistral.
def embed_batch(client: Mistral,texts: list[str]) -> list[list[float]]:
    # Plusieurs tentatives en cas de rate limit.
    for attempt in range(MAX_RETRIES):
        try:
            # Appel API.
            response = client.embeddings.create(
                # Modèle utilisé.
                model=MODEL_NAME,
                # Liste des textes à vectoriser.
                inputs=texts,
            )
            # Extraction des vecteurs.
            return [item.embedding for item in response.data]

        # Capture des erreurs.
        except Exception as exc:
            # Conversion de l'erreur en texte.
            message = str(exc)

            # Si ce n'est pas un rate limit, on relance immédiatement l'erreur.
            if "429" not in message and "rate" not in message.lower():
                raise

            # Calcul d'un temps d'attente exponentiel.
            wait_time = (SLEEP_SECONDS * (2 ** attempt))

            # Message d'information.
            print(
                f"Rate limit détecté. "
                f"Nouvelle tentative dans "
                f"{wait_time:.1f}s..."
            )

            # Pause.
            sleep(wait_time)

    # Si toutes les tentatives échouent.
    raise RuntimeError(
        "Trop d'échecs après rate limit Mistral"
    )


# Fonction principale.
def main() -> None:
    # Lecture de la clé API.
    api_key = os.getenv("MISTRAL_API_KEY")

    # Vérification de la présence de la clé.
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY est absente du fichier .env")

    # Chargement du CSV.
    df = pd.read_csv(PROCESSED_PATH)
    # Suppression des lignes incomplètes.
    df = df.dropna(subset=["uid", "text_for_embedding"])

    # Conversion du DataFrame en liste.
    records = df.to_dict(orient="records")

    # Chargement des embeddings déjà existants.
    existing = load_existing_embeddings()
    # Création de la liste finale.
    embedded_events = list(
        existing.values()
    )

    # Sélection des événements restant à vectoriser.
    records_to_embed = [

        record

        for record in records

        if str(record["uid"])
        not in existing
    ]

    # Informations de suivi.
    print(f"{len(records)} événements au total")

    print(
        f"{len(existing)} événements déjà "
        f"vectorisés avec la stratégie hybride"
    )

    print(
        f"{len(records_to_embed)} événements "
        f"restant à vectoriser"
    )

    print(f"Modèle : {MODEL_NAME}")

    print(
        f"Stratégie d'embedding : "
        f"{EMBEDDING_STRATEGY}"
    )

    print(f"Batch size : {BATCH_SIZE}")

    print(
        f"Pause entre appels : "
        f"{SLEEP_SECONDS}s"
    )

    # Connexion à l'API Mistral.
    with Mistral(api_key=api_key) as client:

        # Traitement par batch.
        for start in range(
            0,
            len(records_to_embed),
            BATCH_SIZE
        ):

            # Extraction du batch courant.
            batch = records_to_embed[
                start : start + BATCH_SIZE
            ]

            # Construction du texte hybride.
            texts = [

                build_hybrid_embedding_text(item)

                for item in batch
            ]

            # Génération des embeddings.
            vectors = embed_batch(
                client,
                texts
            )

            # Association événement ↔ embedding.
            for event, text_used, vector in zip(
                batch,
                texts,
                vectors
            ):

                embedded_events.append(

                    {

                        # Identifiant unique.
                        "uid": str(event["uid"]),

                        # Métadonnées.
                        "title": event["title"],
                        "begin": event["begin"],
                        "venue": event.get("venue"),
                        "city": event.get("city"),
                        "url": event.get("url"),

                        # Texte original.
                        "text_for_embedding":
                            event["text_for_embedding"],

                        # Texte enrichi réellement utilisé.
                        "hybrid_text_for_embedding":
                            text_used,

                        # Informations techniques.
                        "embedding_model":
                            MODEL_NAME,

                        "embedding_strategy":
                            EMBEDDING_STRATEGY,

                        "embedding_dim":
                            len(vector),

                        # Vecteur final.
                        "embedding":
                            vector,
                    }
                )

            # Sauvegarde après chaque batch.
            save_embeddings(
                embedded_events
            )

            # Calcul de la progression.
            done = (
                len(existing)
                + min(
                    start + BATCH_SIZE,
                    len(records_to_embed)
                )
            )

            # Affichage de la progression.
            print(
                f"{done}/{len(records)} vectorisés"
            )

            # Pause entre deux appels API.
            sleep(SLEEP_SECONDS)

    # Résumé final.
    print(
        f"Embeddings sauvegardés dans : "
        f"{EMBEDDINGS_PATH}"
    )

    print(
        f"Nombre total de vecteurs : "
        f"{len(embedded_events)}"
    )

    # Affiche la taille des vecteurs.
    if embedded_events:

        print(
            f"Dimension des vecteurs : "
            f"{embedded_events[0]['embedding_dim']}"
        )


# Vérifie que le script est exécuté directement.
if __name__ == "__main__":

    # Lance le traitement.
    main()