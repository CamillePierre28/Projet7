# Pour savoir précisement pourquoi certains évènements ont été supprimé après le nettoyage
# Les nettoyages appliqués dans preprocess_events() sont :
# df = df.dropna(subset=["uid"])
# df = df.drop_duplicates(subset=["uid"])
# df["begin"] = pd.to_datetime(df["begin"], errors="coerce", utc=True)
# df["end"] = pd.to_datetime(df["end"], errors="coerce", utc=True)
# df = df.dropna(subset=["title", "begin"])
# df = df[df["title"].str.len() > 0]
# df = df[df["text_for_embedding"].str.len() > 80]
# Donc les 2 événements supprimés sont probablement :
# - des doublons de uid
# - ou des événements sans date begin valide
# - ou sans titre
# - ou avec un texte trop court pour être vectorisé

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.preprocessing import normalize_event

RAW_PATH = Path("data/raw/events_raw.json")


def main() -> None:
    raw_events = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    rows = [normalize_event(event) for event in raw_events]
    df = pd.DataFrame(rows)

    print(f"Événements bruts normalisés : {len(df)}")

    missing_uid = df[df["uid"].isna()]
    duplicated_uid = df[df.duplicated(subset=["uid"], keep=False)]
    missing_title = df[df["title"].isna() | (df["title"].str.len() == 0)]

    df["begin_parsed"] = pd.to_datetime(df["begin"], errors="coerce", utc=True)
    missing_begin = df[df["begin_parsed"].isna()]

    short_text = df[df["text_for_embedding"].str.len() <= 80]

    print(f"UID manquants : {len(missing_uid)}")
    print(f"UID dupliqués : {len(duplicated_uid)}")
    print(f"Titres manquants : {len(missing_title)}")
    print(f"Dates begin invalides : {len(missing_begin)}")
    print(f"Textes trop courts : {len(short_text)}")

    removed = pd.concat(
        [
            missing_uid.assign(reason="missing_uid"),
            duplicated_uid.assign(reason="duplicated_uid"),
            missing_title.assign(reason="missing_title"),
            missing_begin.assign(reason="missing_begin"),
            short_text.assign(reason="short_text"),
        ]
    ).drop_duplicates(subset=["uid", "reason"])

    if removed.empty:
        print("Aucun événement problématique identifié individuellement.")
        return

    print("\nÉvénements potentiellement supprimés :")
    print(
        removed[
            [
                "uid",
                "title",
                "begin",
                "city",
                "region",
                "reason",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()