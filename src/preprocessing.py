from __future__ import annotations

import re
from typing import Any

import pandas as pd


def clean_html(text: Any) -> str:
    """
    Nettoie le HTML et normalise les espaces.
    """

    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise un événement OpenDataSoft/OpenAgenda.
    """

    title = clean_html(event.get("title_fr"))

    description = clean_html(event.get("description_fr"))

    long_description = clean_html(
        event.get("longdescription_fr")
    )

    begin = event.get("firstdate_begin")
    end = event.get("firstdate_end")

    city = clean_html(event.get("location_city"))

    region = clean_html(
        event.get("location_region")
    )

    department = clean_html(
        event.get("location_department")
    )

    venue = clean_html(event.get("location_name"))

    address = clean_html(
        event.get("location_address")
    )

    url = event.get("canonicalurl") or ""

    text_for_embedding = "\n".join(
        [
            f"Titre : {title}",
            f"Description : {description}",
            f"Description détaillée : {long_description}",
            f"Lieu : {venue}",
            f"Adresse : {address}",
            f"Ville : {city}",
            f"Département : {department}",
            f"Région : {region}",
            f"Date de début : {begin}",
            f"Date de fin : {end}",
            f"URL : {url}",
        ]
    )

    return {
        "uid": event.get("uid"),
        "title": title,
        "description": description,
        "long_description": long_description,
        "begin": begin,
        "end": end,
        "venue": venue,
        "address": address,
        "city": city,
        "department": department,
        "region": region,
        "url": url,
        "text_for_embedding": clean_html(
            text_for_embedding
        ),
    }


def preprocess_events(
    events: list[dict[str, Any]]
) -> pd.DataFrame:
    """
    Préprocessing principal.
    """

    rows = [
        normalize_event(event)
        for event in events
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Nettoyage UID
    df = df.dropna(subset=["uid"])

    # Suppression doublons
    df = df.drop_duplicates(subset=["uid"])

    # Conversion dates
    df["begin"] = pd.to_datetime(
        df["begin"],
        errors="coerce",
        utc=True,
    )

    df["end"] = pd.to_datetime(
        df["end"],
        errors="coerce",
        utc=True,
    )

    # Suppression lignes invalides
    df = df.dropna(subset=["title", "begin"])

    # Titres non vides
    df = df[
        df["title"].str.len() > 0
    ]

    # Texte suffisamment riche
    df = df[
        df["text_for_embedding"].str.len() > 80
    ]

    # Tri chronologique
    df = df.sort_values("begin")

    return df.reset_index(drop=True)
