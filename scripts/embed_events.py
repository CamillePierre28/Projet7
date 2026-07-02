# Conversion des descriptions des événements en vecteurs numériques grâce au modèle d'embedding de Mistral AI.

from __future__ import annotations

import json
import os
from pathlib import Path
from time import sleep

import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral

# Charge les variables d’environnement du fichier .env.
load_dotenv()

# Chemin du fichier contenant les événements nettoyés.
PROCESSED_PATH = Path("data/processed/events_processed.csv")
# Chemin du fichier où seront sauvegardés les embeddings.
EMBEDDINGS_PATH = Path("data/processed/events_embeddings.json")

# Nom du modèle d’embedding utilisé chez Mistral.
MODEL_NAME = "mistral-embed"
# Nombre d’événements envoyés à l’API en une seule requête. La valeur est configurable dans le .env.
BATCH_SIZE = int(os.getenv("MISTRAL_EMBED_BATCH_SIZE", "8"))
# Temps d’attente entre deux appels API. Cela aide à éviter les rate limits. La valeur est configurable dans le .env aussi.
SLEEP_SECONDS = float(os.getenv("MISTRAL_EMBED_SLEEP_SECONDS", "2"))
# Nombre maximum de tentatives en cas de rate limit API.
MAX_RETRIES = 6

# Fonction qui charge les embeddings déjà existants. Cela évite de recalculer des embeddings déjà créés.
def load_existing_embeddings() -> dict[str, dict]:
    # Si le fichier d’embeddings n’existe pas encore, retourne un dictionnaire vide.
    if not EMBEDDINGS_PATH.exists():
        return {}
    
    # Lit le fichier JSON contenant les embeddings.
    data = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    # Transforme la liste en dictionnaire indexé par uid. Cela permet de retrouver rapidement un événement déjà vectorisé.
    return {str(item["uid"]): item for item in data}

# Fonction qui sauvegarde les embeddings dans un fichier JSON.
def save_embeddings(items: list[dict]) -> None:
    # Crée automatiquement le dossier parent si nécessaire.
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Sauvegarde les embeddings au format JSON.
    EMBEDDINGS_PATH.write_text(
        # Convertit les données Python en JSON lisible.
        json.dumps(items, ensure_ascii=False, indent=2),
        # Encode le fichier en UTF-8.
        encoding="utf-8",
    )

# Fonction qui envoie un batch de textes à Mistral afin de générer leurs embeddings.
def embed_batch(client: Mistral, texts: list[str]) -> list[list[float]]:
    # Réessaie plusieurs fois si l’API retourne un rate limit.
    for attempt in range(MAX_RETRIES):
        try:
            # Appel API vers le modèle d’embedding Mistral.
            response = client.embeddings.create(
                # Nom du modèle utilisé.
                model=MODEL_NAME,
                # Liste des textes à vectoriser.
                inputs=texts,
            )
            # Retourne uniquement les vecteurs numériques.
            return [item.embedding for item in response.data]

        # Capture toutes les erreurs potentielles.
        except Exception as exc:
            # Convertit l’erreur en texte.
            message = str(exc)

            # Si l’erreur n’est pas liée à un rate limit, on relance immédiatement l’erreur.
            if "429" not in message and "rate" not in message.lower():
                raise

            # Calcule un temps d’attente exponentiel. Exemple : 2s → 4s → 8s → 16s...
            wait_time = SLEEP_SECONDS * (2 ** attempt)

            # Affiche un message d’attente.
            print(
                f"Rate limit détecté. Nouvelle tentative dans "
                f"{wait_time:.1f}s..."
            )

            # Met le programme en pause avant de réessayer.
            sleep(wait_time)

    # Si toutes les tentatives échouent, une erreur finale est déclenchée.
    raise RuntimeError("Trop d'échecs après rate limit Mistral")

# Fonction principale du script.
def main() -> None:
    # Récupère la clé API Mistral depuis le fichier .env.
    api_key = os.getenv("MISTRAL_API_KEY")

    # Vérifie que la clé API existe.
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY est absente du fichier .env")

    # Charge les événements nettoyés depuis le CSV.
    df = pd.read_csv(PROCESSED_PATH)
    # Supprime les lignes sans uid ou sans texte à vectoriser.
    df = df.dropna(subset=["uid", "text_for_embedding"])

    # Convertit le DataFrame en liste de dictionnaires.
    records = df.to_dict(orient="records")

    # Charge les embeddings déjà existants.
    existing = load_existing_embeddings()
    # Initialise la liste des événements déjà vectorisés.
    embedded_events = list(existing.values())

    # Sélectionne uniquement les événements qui n’ont pas encore d’embedding.
    records_to_embed = [
        record for record in records
        if str(record["uid"]) not in existing
    ]

    # Affiche le nombre total d’événements.
    print(f"{len(records)} événements au total")
    # Affiche combien sont déjà vectorisés.
    print(f"{len(existing)} événements déjà vectorisés")
    # Affiche combien restent à vectoriser.
    print(f"{len(records_to_embed)} événements restant à vectoriser")
    # Affiche la taille des batches.
    print(f"Batch size : {BATCH_SIZE}")
    # Affiche la pause entre les appels API.
    print(f"Pause entre appels : {SLEEP_SECONDS}s")

    # Ouvre une connexion avec l’API Mistral.
    with Mistral(api_key=api_key) as client:
        # Parcourt les événements par groupes de taille BATCH_SIZE.
        for start in range(0, len(records_to_embed), BATCH_SIZE):
            # Sélectionne un batch d’événements.
            batch = records_to_embed[start : start + BATCH_SIZE]
            # Extrait uniquement les textes à vectoriser.
            texts = [item["text_for_embedding"] for item in batch]

            # Génère les embeddings du batch.
            vectors = embed_batch(client, texts)

            # Associe chaque événement à son vecteur.
            for event, vector in zip(batch, vectors):
                # Ajoute l’événement enrichi avec son embedding.
                embedded_events.append(
                    {
                        # Identifiant unique.
                        "uid": str(event["uid"]), 
                        # Métadonnées de l’événement.
                        "title": event["title"],
                        "begin": event["begin"],
                        "end": event.get("end"),
                        "venue": event.get("venue"),
                        "address": event.get("address"),
                        "city": event.get("city"),
                        "department": event.get("department"),
                        "region": event.get("region"),
                        "url": event.get("url"),
                        # Texte ayant servi à créer l’embedding
                        "text_for_embedding": event["text_for_embedding"],
                        # Nom du modèle utilisé.
                        "embedding_model": MODEL_NAME,
                        # Taille du vecteur.
                        "embedding_dim": len(vector),
                        # Le vecteur numérique lui-même.
                        "embedding": vector,
                    }
                )

            # Sauvegarde progressive des embeddings. Cela évite de tout perdre si le script s’arrête.
            save_embeddings(embedded_events)

            # Calcule combien d’événements sont terminés.
            done = len(existing) + min(start + BATCH_SIZE, len(records_to_embed))
            # Affiche la progression.
            print(f"{done}/{len(records)} vectorisés")

            # Pause entre deux appels API.
            sleep(SLEEP_SECONDS)

    # Affiche le chemin du fichier généré.
    print(f"Embeddings sauvegardés dans : {EMBEDDINGS_PATH}")
    # Affiche le nombre total de vecteurs générés.
    print(f"Nombre total de vecteurs : {len(embedded_events)}")

    # Vérifie qu’il existe au moins un embedding.
    if embedded_events:
        # Affiche la dimension du vecteur.
        print(f"Dimension des vecteurs : {embedded_events[0]['embedding_dim']}")


# Vérifie que le fichier est exécuté directement.
if __name__ == "__main__":
    # Lance la fonction principale.
    main()








# Approche classique (métadonnées séparées) : vectorise uniquement text_for_embedding et stocke à côté "title": "...", "city": "...", "region": "...", "begin": "...".


# Explication du script

# Ce script sert à transformer les descriptions des événements en vecteurs numériques, appelés embeddings. Ces vecteurs permettent ensuite d’effectuer une recherche sémantique dans le chatbot RAG.
# Autrement dit :
# - un utilisateur écrit une question 
# - la question est transformée en vecteur 
# - le système compare ce vecteur avec ceux des événements 
# - il retrouve les événements les plus proches en sens

# Qu’est-ce qu’un embedding ?
# Un embedding est une représentation mathématique d’un texte. Par exemple : "concert jazz à Toulouse" devient un long vecteur de nombres : [0.182, -0.044, 0.921, ...]
# Le modèle encode le sens du texte dans l’espace vectoriel. Des textes proches sémantiquement auront des vecteurs proches.

# Pourquoi utiliser mistral-embed ?

# Le modèle : MODEL_NAME = "mistral-embed" est spécialisé dans la génération d’embeddings. C’est un bon choix parce qu’il est :
# - performant pour la recherche sémantique
# - compatible avec les pipelines RAG 
# - optimisé pour comparer des textes 
# - plus léger qu’un LLM complet

# Pourquoi vectoriser par batch ? 
# Au lieu d’envoyer un texte à la fois : inputs=[texte] le script envoie plusieurs textes d’un coup : inputs=texts. Cela :
# - réduit le nombre d’appels API
# - accélère le traitement
# - diminue les coûts
# - améliore les performances

# Pourquoi gérer les rate limits ?

# Les API limitent souvent le nombre de requêtes. Le script détecte donc les erreurs :
# - 429
# - rate limit
# Puis il attend progressivement plus longtemps avant de réessayer : 2s → 4s → 8s → 16s. C’est ce qu’on appelle un : exponential backoff. C’est une très bonne pratique en production.

# Pourquoi sauvegarder progressivement ?

# Après chaque batch : save_embeddings(embedded_events). Cela évite de perdre tout le travail si :
# - le script plante 
# - la connexion coupe 
# - le quota API est dépassé 
# - l’ordinateur s’éteint

# Pourquoi recharger les embeddings existants ?

# Cette partie : existing = load_existing_embeddings() permet de reprendre le travail là où il s’était arrêté. Le script évite ainsi de recalculer des embeddings déjà générés.
# C’est très utile quand :
# - il y a beaucoup d’événements 
# - l’API coûte de l’argent 
# - le traitement est long

# Pourquoi stocker les métadonnées avec les vecteurs ? 

# Le fichier final ne contient pas seulement l’embedding : "embedding": vector, mais aussi :
# - "title"
# - "city"
# - "begin"
# - "url"
# Cela permet ensuite :
# - d’afficher les résultats 
# - de filtrer les événements 
# - de reconstruire facilement le contexte pour le chatbot

# Pourquoi utiliser JSON plutôt qu’un CSV ?

# Les embeddings sont des listes de centaines de nombres : [0.124, -0.882, ...]. Le JSON gère beaucoup mieux les structures complexes que le CSV.