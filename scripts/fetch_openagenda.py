# Récupération des données d'événements à partir de la plateforme Open Agenda

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.preprocessing import filter_recent_and_upcoming_events, preprocess_events

# Charge les variables d'environnement définies dans le fichier .env.
load_dotenv()

# Récupère l'URL de l'API OpenAgenda depuis le fichier .env.
API_URL = os.getenv("OPENAGENDA_API_URL")
# Récupère le nombre maximum de pages à appeler dans l'API. Si la variable n'existe pas, la valeur par défaut est 50.
EVENT_MAX_PAGES = int(os.getenv("EVENT_MAX_PAGES", "50"))
# Récupère le nombre de jours à conserver dans le passé pour les événements récents. Si la variable n'existe pas, la valeur par défaut est 365 jours.
EVENT_LOOKBACK_DAYS = int(os.getenv("EVENT_LOOKBACK_DAYS", "365"))

# Chemin du fichier où seront sauvegardées les données brutes récupérées depuis l'API.
RAW_PATH = Path("data/raw/events_raw.json")
# Chemin du fichier CSV où seront sauvegardées les données nettoyées.
PROCESSED_CSV_PATH = Path("data/processed/events_processed.csv")
# Chemin du fichier JSON où seront sauvegardées les données nettoyées.
PROCESSED_JSON_PATH = Path("data/processed/events_processed.json")


# Fonction qui récupère les événements depuis l'API, limit correspond au nombre d'événements récupérés par requête, max_pages correspond au nombre maximum de pages à parcourir.
def fetch_events(limit: int = 100, max_pages: int = EVENT_MAX_PAGES) -> list[dict]:
    # Vérifie que l'URL de l'API est bien présente dans le fichier .env. Sans cette URL, le script ne peut pas récupérer les données.
    if not API_URL:
        raise RuntimeError("OPENAGENDA_API_URL est absente du fichier .env")
    
    # Initialise une liste vide qui contiendra tous les événements récupérés.
    events = []

    # Boucle sur les pages de résultats de l'API. Cela permet de récupérer plus que les 100 premiers événements.
    for page in range(max_pages):
        # Calcule le décalage à appliquer dans l'API. Exemple : page 0 = offset 0, page 1 = offset 100, page 2 = offset 200.
        offset = page * limit

        # Paramètres envoyés à l'API.
        params = {
            # Indique à partir de quel événement commencer la récupération.
            "offset": offset,
            # Trie les événements par date de début décroissante, donc les événements les plus récents arrivent en premier.
            "order_by": "firstdate_begin desc",
        }

        # Envoie une requête GET à l'API, timeout=30 évite que le programme reste bloqué trop longtemps.
        response = requests.get(API_URL, params=params, timeout=30)
        # Déclenche une erreur si la réponse HTTP indique un problème. Exemple : 404, 500, etc.
        response.raise_for_status()

        # Convertit la réponse JSON de l'API en dictionnaire Python.
        payload = response.json()
        # Récupère la liste des événements dans la clé "results". Si la clé n'existe pas, on utilise une liste vide par sécurité.
        results = payload.get("results", [])

        # Si aucun résultat n'est retourné, cela signifie qu'il n'y a plus de données. On arrête donc la boucle.
        if not results:
            break

        # Ajoute les événements récupérés à la liste globale.
        events.extend(results)

        # Affiche le nombre d'événements récupérés jusqu'à maintenant.
        print(f"{len(events)} événements bruts récupérés...")

        # Si le nombre de résultats est inférieur à la limite demandée, cela signifie qu'on est arrivé à la dernière page.
        if len(results) < limit:
            break

    # Retourne la liste complète des événements récupérés.
    return events


# Fonction principale du script. Elle orchestre la récupération, le nettoyage, le filtrage et la sauvegarde.
def main() -> None:
    # Crée le dossier data/raw si celui-ci n'existe pas encore, parents=True crée aussi les dossiers parents nécessaires, exist_ok=True évite une erreur si le dossier existe déjà.
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Crée le dossier data/processed si celui-ci n'existe pas encore.
    PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Affiche un message pour indiquer le début de la récupération.
    print("Récupération des événements OpenDataSoft / OpenAgenda...")
    # Appelle la fonction fetch_events pour récupérer les événements depuis l'API.
    events = fetch_events(limit=100, max_pages=EVENT_MAX_PAGES)

    # Affiche le nombre total d'événements récupérés avant traitement.
    print(f"{len(events)} événements bruts récupérés au total")

    # Sauvegarde les données brutes dans un fichier JSON.
    RAW_PATH.write_text(
        # Convertit la liste d'événements Python en texte JSON, ensure_ascii=False permet de conserver les accents, indent=2 rend le fichier lisible.
        json.dumps(events, ensure_ascii=False, indent=2),
        # Encode le fichier en UTF-8 pour bien gérer les caractères français.
        encoding="utf-8",
    )

    # Nettoie et transforme les événements bruts en DataFrame pandas.
    df = preprocess_events(events)

    # Affiche le nombre d'événements conservés après le nettoyage initial.
    print(f"{len(df)} événements conservés après nettoyage initial")

    # Filtre les événements pour conserver : les événements récents et les événements à venir.
    df = filter_recent_and_upcoming_events(
        df,
        # Nombre de jours dans le passé à conserver.
        lookback_days=EVENT_LOOKBACK_DAYS,
    )

    # Affiche le nombre d'événements restants après le filtrage temporel.
    print(
        f"{len(df)} événements conservés après filtrage temporel "
        f"sur les {EVENT_LOOKBACK_DAYS} derniers jours et les événements à venir"
    )

    # Sauvegarde les données nettoyées au format CSV, index=False évite d'ajouter l'index pandas comme colonne inutile.
    df.to_csv(PROCESSED_CSV_PATH, index=False)

    # Sauvegarde aussi les données nettoyées au format JSON.
    df.to_json(
        PROCESSED_JSON_PATH,
        # Chaque ligne du JSON correspond à un événement.
        orient="records",
        # Conserve les accents dans le fichier JSON.
        force_ascii=False,
        # Rend le JSON plus lisible.
        indent=2,
        # Formate les dates au format ISO.
        date_format="iso",
    )

    # Affiche le chemin du fichier CSV sauvegardé.
    print(f"Données sauvegardées : {PROCESSED_CSV_PATH}")
    # Affiche le chemin du fichier JSON sauvegardé.
    print(f"Données sauvegardées : {PROCESSED_JSON_PATH}")


# Cette condition vérifie que le fichier est exécuté directement. Si le fichier est importé dans un autre script, main() ne sera pas lancé automatiquement.
if __name__ == "__main__":
    # Lance la fonction principale.
    main()


# Explication du script

# Ce script sert à récupérer des événements depuis l’API OpenAgenda/OpenDataSoft, à les sauvegarder en brut, puis à les nettoyer, filtrer et exporter dans des fichiers exploitables pour la suite du projet.

# Il joue donc le rôle de script d’extraction et de préparation des données.

# Dans ce projet, il sert d’étape avant la création d’un chatbot RAG. Il permet d’obtenir un jeu de données propre à partir de données publiques.

# Pourquoi utiliser requests ?

# requests est utilisé pour faire des appels HTTP vers l’API. Ici, il permet d’envoyer une requête GET à OpenAgenda et de récupérer les événements au format JSON.

# C’est une bibliothèque simple, lisible et très utilisée pour interroger des API.

# Pourquoi utiliser pandas ?

# pandas est utilisé parce que les événements deviennent ensuite des données structurées en tableau. Cela facilite le nettoyage, le filtrage, la suppression de colonnes inutiles, la gestion des dates et l’export en CSV ou JSON.

# Pourquoi sauvegarder en JSON brut ?

# Le fichier : data/raw/events_raw.json sert à garder une copie exacte des données récupérées depuis l’API, avant nettoyage. C’est utile pour déboguer, comparer ou relancer le traitement sans refaire l’appel API.

# Pourquoi sauvegarder en CSV et JSON traités ?

# Le CSV est pratique pour ouvrir les données dans Excel, pandas ou un tableur. Le JSON est pratique pour être réutilisé dans une application, une API, un moteur de recherche ou une étape d’indexation vectorielle.

# Pourquoi faire une pagination avec offset ?

# L’API ne renvoie pas tous les événements d’un coup. Le script récupère donc les données page par page avec : offset = page * limit. Cela permet de récupérer progressivement 100 événements par requête.

# Pourquoi filtrer les événements dans le temps ?

# La fonction : filter_recent_and_upcoming_events(...) évite de garder des événements trop anciens. C’est important pour ton chatbot, car il doit recommander ou retrouver des événements encore pertinents.