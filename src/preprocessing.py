# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe re pour utiliser des expressions regulieres, utiles pour nettoyer le HTML.
import re
# Importe Any pour indiquer qu'une valeur peut etre de n'importe quel type.
from typing import Any

# Importe pandas pour manipuler les evenements sous forme de tableau.
import pandas as pd


# Nettoie un texte qui peut contenir du HTML ou des espaces inutiles.
def clean_html(text: Any) -> str:
    """
    Nettoie le HTML et normalise les espaces.
    """

    # Si la valeur est None, on retourne une chaine vide pour eviter une erreur.
    if text is None:
        # Retourne un texte vide quand il n'y a rien a nettoyer.
        return ""

    # Convertit la valeur en texte, meme si elle est d'un autre type au depart.
    text = str(text)

    # Supprime toutes les balises HTML, par exemple <p> ou <strong>.
    text = re.sub(r"<[^>]+>", " ", text)
    # Remplace plusieurs espaces, retours ligne ou tabulations par un seul espace.
    text = re.sub(r"\s+", " ", text)

    # Supprime les espaces au debut et a la fin du texte.
    return text.strip()


# Transforme un evenement brut OpenAgenda en dictionnaire propre et standardise.
def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise un événement OpenDataSoft/OpenAgenda.
    """

    # Recupere et nettoie le titre francais de l'evenement.
    title = clean_html(event.get("title_fr"))

    # Recupere et nettoie la description courte.
    description = clean_html(event.get("description_fr"))

    # Recupere et nettoie la description longue.
    long_description = clean_html(
        event.get("longdescription_fr")
    )

    # Recupere la date de debut brute.
    begin = event.get("firstdate_begin")
    # Recupere la date de fin brute.
    end = event.get("firstdate_end")

    # Recupere et nettoie la ville de l'evenement.
    city = clean_html(event.get("location_city"))

    # Recupere et nettoie la region.
    region = clean_html(
        event.get("location_region")
    )

    # Recupere et nettoie le departement.
    department = clean_html(
        event.get("location_department")
    )

    # Recupere et nettoie le nom du lieu.
    venue = clean_html(event.get("location_name"))

    # Recupere et nettoie l'adresse.
    address = clean_html(
        event.get("location_address")
    )

    # Recupere l'URL officielle de l'evenement, ou une chaine vide si elle manque.
    url = event.get("canonicalurl") or ""

    # Construit un texte complet qui servira a creer l'embedding de l'evenement.
    text_for_embedding = "\n".join(
        [
            # Ajoute le titre dans le texte final.
            f"Titre : {title}",
            # Ajoute la description courte.
            f"Description : {description}",
            # Ajoute la description longue.
            f"Description détaillée : {long_description}",
            # Ajoute le lieu.
            f"Lieu : {venue}",
            # Ajoute l'adresse.
            f"Adresse : {address}",
            # Ajoute la ville.
            f"Ville : {city}",
            # Ajoute le departement.
            f"Département : {department}",
            # Ajoute la region.
            f"Région : {region}",
            # Ajoute la date de debut.
            f"Date de début : {begin}",
            # Ajoute la date de fin.
            f"Date de fin : {end}",
            # Ajoute l'URL source.
            f"URL : {url}",
        ]
    )

    # Retourne un evenement propre avec des noms de champs simples.
    return {
        # Conserve l'identifiant unique de l'evenement.
        "uid": event.get("uid"),
        # Stocke le titre nettoye.
        "title": title,
        # Stocke la description courte nettoyee.
        "description": description,
        # Stocke la description longue nettoyee.
        "long_description": long_description,
        # Stocke la date de debut brute pour conversion plus tard.
        "begin": begin,
        # Stocke la date de fin brute pour conversion plus tard.
        "end": end,
        # Stocke le lieu.
        "venue": venue,
        # Stocke l'adresse.
        "address": address,
        # Stocke la ville.
        "city": city,
        # Stocke le departement.
        "department": department,
        # Stocke la region.
        "region": region,
        # Stocke l'URL.
        "url": url,
        # Stocke le texte complet nettoye qui sera utilise pour les embeddings.
        "text_for_embedding": clean_html(
            text_for_embedding
        ),
    }


# Nettoie une liste complete d'evenements et la transforme en DataFrame pandas.
def preprocess_events(
    events: list[dict[str, Any]]
) -> pd.DataFrame:
    """
    Préprocessing principal.
    """

    # Normalise chaque evenement brut de la liste.
    rows = [
        # Transforme un evenement brut en evenement propre.
        normalize_event(event)
        # Parcourt tous les evenements recus.
        for event in events
    ]

    # Transforme la liste de dictionnaires en tableau pandas.
    df = pd.DataFrame(rows)

    # Si le tableau est vide, on le retourne directement.
    if df.empty:
        # Retourne le DataFrame vide sans appliquer d'autres traitements.
        return df

    # Nettoyage UID
    # Supprime les lignes sans identifiant unique.
    df = df.dropna(subset=["uid"])

    # Suppression doublons
    # Supprime les evenements dupliques qui ont le meme UID.
    df = df.drop_duplicates(subset=["uid"])

    # Conversion dates
    # Convertit la date de debut en vraie date pandas, en UTC.
    df["begin"] = pd.to_datetime(
        # Colonne a convertir.
        df["begin"],
        # Les dates invalides deviennent NaT au lieu de provoquer une erreur.
        errors="coerce",
        # Force les dates en fuseau UTC.
        utc=True,
    )

    # Convertit la date de fin en vraie date pandas, en UTC.
    df["end"] = pd.to_datetime(
        # Colonne a convertir.
        df["end"],
        # Les dates invalides deviennent NaT.
        errors="coerce",
        # Force les dates en fuseau UTC.
        utc=True,
    )

    # Suppression lignes invalides
    # Supprime les lignes sans titre ou sans date de debut valide.
    df = df.dropna(subset=["title", "begin"])

    # Titres non vides
    # Garde uniquement les evenements dont le titre contient au moins un caractere.
    df = df[
        df["title"].str.len() > 0
    ]

    # Texte suffisamment riche
    # Garde uniquement les evenements avec assez de texte pour etre utiles au RAG.
    df = df[
        df["text_for_embedding"].str.len() > 80
    ]

    # Tri chronologique
    # Trie les evenements du plus ancien au plus recent selon la date de debut.
    df = df.sort_values("begin")

    # Reinitialise l'index du tableau avant de le retourner.
    return df.reset_index(drop=True)


# Filtre le corpus pour garder les evenements recents et futurs.
def filter_recent_and_upcoming_events(
    df: pd.DataFrame,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """
    Conserve les événements :
    - situés dans les X derniers jours
    - ou à venir

    Cela permet d'avoir un corpus récent incluant l'historique proche
    et les événements futurs.
    """

    # Si le DataFrame est vide, il n'y a rien a filtrer.
    if df.empty:
        # Retourne le DataFrame vide tel quel.
        return df

    # Recupere la date actuelle en UTC.
    now = pd.Timestamp.now(tz="UTC")
    # Calcule la date minimale a conserver dans le passe.
    min_date = now - pd.Timedelta(days=lookback_days)

    # Garde les evenements dont la date de debut est apres la date minimale.
    filtered = df[df["begin"] >= min_date]

    # Trie le resultat par date puis reinitialise l'index.
    return filtered.sort_values("begin").reset_index(drop=True)