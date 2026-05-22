# Converion des descriptions des événements en format vectoriel avec le modèle de NLP Mistral

from __future__ import annotations

import json
import os
from pathlib import Path
from time import sleep

import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

PROCESSED_PATH = Path("data/processed/events_processed.csv")
EMBEDDINGS_PATH = Path("data/processed/events_embeddings.json")

MODEL_NAME = "mistral-embed"
BATCH_SIZE = 32


def embed_batch(client: Mistral, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=MODEL_NAME,
        inputs=texts,
    )

    return [item.embedding for item in response.data]


def main() -> None:
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY est absente du fichier .env")

    df = pd.read_csv(PROCESSED_PATH)

    df = df.dropna(subset=["uid", "text_for_embedding"])
    records = df.to_dict(orient="records")

    print(f"{len(records)} événements à vectoriser avec {MODEL_NAME}")

    embedded_events = []

    with Mistral(api_key=api_key) as client:
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            texts = [item["text_for_embedding"] for item in batch]

            vectors = embed_batch(client, texts)

            for event, vector in zip(batch, vectors):
                embedded_events.append(
                    {
                        "uid": event["uid"],
                        "title": event["title"],
                        "begin": event["begin"],
                        "end": event.get("end"),
                        "venue": event.get("venue"),
                        "address": event.get("address"),
                        "city": event.get("city"),
                        "department": event.get("department"),
                        "region": event.get("region"),
                        "url": event.get("url"),
                        "text_for_embedding": event["text_for_embedding"],
                        "embedding_model": MODEL_NAME,
                        "embedding_dim": len(vector),
                        "embedding": vector,
                    }
                )

            print(f"{min(start + BATCH_SIZE, len(records))}/{len(records)} vectorisés")

            sleep(0.2)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    EMBEDDINGS_PATH.write_text(
        json.dumps(embedded_events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Embeddings sauvegardés dans : {EMBEDDINGS_PATH}")
    print(f"Nombre total de vecteurs : {len(embedded_events)}")
    print(f"Dimension des vecteurs : {embedded_events[0]['embedding_dim']}")


if __name__ == "__main__":
    main()