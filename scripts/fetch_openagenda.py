from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from src.preprocessing import preprocess_events

API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/"
    "datasets/evenements-publics-openagenda/records"
)

RAW_PATH = Path("data/raw/events_raw.json")
PROCESSED_CSV_PATH = Path("data/processed/events_processed.csv")
PROCESSED_JSON_PATH = Path("data/processed/events_processed.json")


def fetch_events(limit: int = 100, max_pages: int = 10) -> list[dict]:
    events = []

    for page in range(max_pages):
        offset = page * limit

        params = {
            "limit": limit,
            "offset": offset,
            "refine": [
                'location_region:"Occitanie"',
                'firstdate_begin:"2025"',
            ],
        }

        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json()
        results = payload.get("results", [])

        if not results:
            break

        events.extend(results)

        if len(results) < limit:
            break

    return events


def main() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Récupération des événements OpenDataSoft / OpenAgenda...")
    events = fetch_events(limit=100, max_pages=10)

    print(f"{len(events)} événements bruts récupérés")

    RAW_PATH.write_text(
        json.dumps(events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    df = preprocess_events(events)

    print(f"{len(df)} événements conservés après nettoyage")

    df.to_csv(PROCESSED_CSV_PATH, index=False)
    df.to_json(PROCESSED_JSON_PATH, orient="records", force_ascii=False, indent=2, date_format="iso")

    print(f"Données sauvegardées : {PROCESSED_CSV_PATH}")
    print(f"Données sauvegardées : {PROCESSED_JSON_PATH}")


if __name__ == "__main__":
    main()