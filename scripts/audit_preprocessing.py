# Pour savoir précisément pourquoi certains événements ont été supprimés après le nettoyage.
# Ce script sert donc à diagnostiquer les pertes de données entre les données brutes et les données nettoyées.

# Les nettoyages appliqués dans preprocess_events() sont :
# Suppression des événements sans uid : df = df.dropna(subset=["uid"])
# Suppression des événements avec un uid déjà présent : df = df.drop_duplicates(subset=["uid"])
# Conversion de la date de début en vraie date pandas : df["begin"] = pd.to_datetime(df["begin"], errors="coerce", utc=True)
# Conversion de la date de fin en vraie date pandas : df["end"] = pd.to_datetime(df["end"], errors="coerce", utc=True)
# Suppression des événements sans titre ou sans date de début valide : df = df.dropna(subset=["title", "begin"])
# Suppression des événements dont le titre est vide : df = df[df["title"].str.len() > 0]
# Suppression des événements dont le texte à vectoriser est trop court : df = df[df["text_for_embedding"].str.len() > 80]

# Les événements supprimés sont donc probablement :
# - des doublons de uid ;
# - des événements sans date begin valide ;
# - des événements sans titre ;
# - des événements avec un texte trop court pour être vectorisé.


from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.preprocessing import normalize_event

# Chemin du fichier contenant les événements bruts récupérés depuis l’API.
RAW_PATH = Path("data/raw/events_raw.json")

# Fonction principale du script.
def main() -> None:
    # Lit le fichier JSON brut et le transforme en objet Python. Ici, raw_events devient une liste de dictionnaires.
    raw_events = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    # Applique normalize_event à chaque événement brut. Cela permet d’obtenir des événements avec une structure uniforme.
    rows = [normalize_event(event) for event in raw_events]
    # Transforme la liste d’événements normalisés en DataFrame pandas.
    df = pd.DataFrame(rows)

    # Affiche le nombre total d’événements après normalisation.
    print(f"Événements bruts normalisés : {len(df)}")

    # Sélectionne les lignes où l’uid est manquant. L’uid sert d’identifiant unique pour chaque événement.
    missing_uid = df[df["uid"].isna()]
    # Sélectionne tous les événements dont l’uid apparaît plusieurs fois, keep=False permet de marquer toutes les occurrences du doublon.
    duplicated_uid = df[df.duplicated(subset=["uid"], keep=False)]
    # Sélectionne les événements sans titre ou avec un titre vide.
    missing_title = df[df["title"].isna() | (df["title"].str.len() == 0)]

    # Convertit la colonne begin en vraie date pandas, errors="coerce" transforme les dates invalides en NaT, utc=True force les dates à être interprétées en UTC.
    df["begin_parsed"] = pd.to_datetime(df["begin"], errors="coerce", utc=True)
    # Sélectionne les événements dont la date begin n’a pas pu être convertie.
    missing_begin = df[df["begin_parsed"].isna()]

    # Sélectionne les événements dont le texte utilisé pour l’embedding est trop court. Un texte trop court est peu utile pour une recherche sémantique.
    short_text = df[df["text_for_embedding"].str.len() <= 80]

    # Affiche le nombre d’événements sans uid.
    print(f"UID manquants : {len(missing_uid)}")
    # Affiche le nombre d’événements ayant un uid dupliqué.
    print(f"UID dupliqués : {len(duplicated_uid)}")
    # Affiche le nombre d’événements sans titre.
    print(f"Titres manquants : {len(missing_title)}")
    # Affiche le nombre d’événements avec une date begin invalide.
    print(f"Dates begin invalides : {len(missing_begin)}")
    # Affiche le nombre d’événements avec un texte trop court.
    print(f"Textes trop courts : {len(short_text)}")

    # Regroupe tous les événements problématiques dans un seul DataFrame.
    removed = pd.concat(
        [
            # Ajoute la raison "missing_uid" aux événements sans uid.
            missing_uid.assign(reason="missing_uid"),
            # Ajoute la raison "duplicated_uid" aux événements dupliqués.
            duplicated_uid.assign(reason="duplicated_uid"),
            # Ajoute la raison "missing_title" aux événements sans titre.
            missing_title.assign(reason="missing_title"),
            # Ajoute la raison "missing_begin" aux événements sans date valide.
            missing_begin.assign(reason="missing_begin"),
            # Ajoute la raison "short_text" aux événements avec texte trop court.
            short_text.assign(reason="short_text"),
        ]

    # Supprime les doublons exacts basés sur le couple uid + reason. Cela évite d’afficher plusieurs fois le même problème pour le même événement.
    ).drop_duplicates(subset=["uid", "reason"])

    # Si aucun événement problématique n’a été trouvé, on affiche un message puis on arrête le script.
    if removed.empty:
        print("Aucun événement problématique identifié individuellement.")
        return

    # Affiche un titre avant de lister les événements potentiellement supprimés.
    print("\nÉvénements potentiellement supprimés :")
    # Affiche uniquement les colonnes utiles pour comprendre les suppressions.
    print(
        removed[
            [
                "uid",          # Identifiant unique de l’événement.
                "title",        # Titre de l’événement.
                "begin",        # Date de début brute.
                "city",         # Ville de l’événement.
                "region",       # Région de l’événement.
                "reason",       # Raison pour laquelle l’événement est problématique.
            ]
        # Convertit le DataFrame en texte lisible dans le terminal, index=False évite d’afficher l’index pandas.
        ].to_string(index=False)
    )

# Vérifie que le script est lancé directement. Si le fichier est importé ailleurs, main() ne s’exécute pas automatiquement.
if __name__ == "__main__":
    # Lance la fonction principale.
    main()








# Explication du script

# Ce script sert à comprendre pourquoi certains événements disparaissent après le nettoyage. Il ne récupère pas les données depuis l’API. Il part du fichier brut déjà sauvegardé : data/raw/events_raw.json
# Puis il applique seulement la normalisation avec : normalize_event(event) Ensuite, il vérifie les principales raisons possibles de suppression :
# - missing_uid
# - duplicated_uid
# - missing_title
# - missing_begin
# - short_text

# Il sert de script de diagnostic ou de débogage. Quand on voit par exemple qu'on avait 5000 événements bruts mais seulement 4998 après nettoyage, ce script permet d’identifier quels événements ont été supprimés et pourquoi.
# C’est très utile pour justifier les pertes de données dans ce projet.

# Pourquoi utiliser normalize_event ?

# normalize_event permet de transformer chaque événement brut en un format commun. Les données venant de l’API peuvent être complexes, imbriquées ou irrégulières. La normalisation permet d’obtenir des colonnes simples comme :
# - uid
# - title
# - begin
# - city
# - region
# - text_for_embedding
# Cela rend ensuite l’analyse beaucoup plus facile avec pandas.

# Pourquoi vérifier les doublons d’UID ?

# L’uid représente l’identifiant unique d’un événement. Si deux événements ont le même uid, ils sont considérés comme des doublons. Dans le nettoyage principal, ils sont supprimés pour éviter d’avoir plusieurs fois le même événement dans la base.

# Pourquoi vérifier les dates invalides ?

# Un événement sans date de début valide est difficile à exploiter. Dans ce projet, la date est importante pour savoir si un événement est récent, passé ou à venir. C’est donc logique de supprimer les événements dont begin est invalide.

# Pourquoi vérifier les textes trop courts ?

# Le champ : text_for_embedding sert à créer les embeddings pour la recherche vectorielle. Un texte trop court donne peu de contexte au modèle d’embedding. Il risque donc d’être peu utile pour retrouver correctement l’événement dans un chatbot RAG.

# Pourquoi utiliser pd.concat ?

# pd.concat permet de regrouper tous les problèmes détectés dans une seule table. Chaque événement problématique reçoit une colonne : reason. Cela permet de savoir clairement pourquoi il a pu être supprimé.