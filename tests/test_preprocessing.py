import pandas as pd

from src.preprocessing import (
    clean_html,
    normalize_event,
    preprocess_events,
)


def test_clean_html_removes_tags():
    text = "<p>Concert <strong>gratuit</strong></p>"
    assert clean_html(text) == "Concert gratuit"


def test_normalize_event_returns_expected_fields():
    event = {
        "uid": "123",
        "title_fr": "Exposition Monet",
        "description_fr": "Une belle exposition",
        "longdescription_fr": "<p>Description détaillée</p>",
        "canonicalurl": "https://example.com/event",
        "firstdate_begin": "2025-06-01T10:00:00+00:00",
        "firstdate_end": "2025-06-01T18:00:00+00:00",
        "location_name": "Musée Test",
        "location_address": "1 rue Exemple",
        "location_city": "Toulouse",
        "location_department": "Haute-Garonne",
        "location_region": "Occitanie",
    }

    normalized = normalize_event(event)

    assert normalized["uid"] == "123"
    assert normalized["title"] == "Exposition Monet"
    assert normalized["city"] == "Toulouse"
    assert normalized["region"] == "Occitanie"
    assert "Exposition Monet" in normalized["text_for_embedding"]


def test_preprocess_events_drops_invalid_events():
    events = [
        {
            "uid": "1",
            "title_fr": "Concert valide",
            "description_fr": "Description suffisamment longue pour être indexée dans le futur système RAG.",
            "longdescription_fr": "Description longue également exploitable.",
            "firstdate_begin": "2025-06-01T10:00:00+00:00",
            "firstdate_end": "2025-06-01T12:00:00+00:00",
            "location_city": "Toulouse",
            "location_region": "Occitanie",
        },
        {
            "uid": None,
            "title_fr": "Sans UID",
            "firstdate_begin": "2025-06-01T10:00:00+00:00",
            "location_city": "Toulouse",
            "location_region": "Occitanie",
        },
    ]

    df = preprocess_events(events)

    assert len(df) == 1
    assert df.iloc[0]["uid"] == "1"


def test_preprocess_events_converts_dates():
    events = [
        {
            "uid": "1",
            "title_fr": "Événement test",
            "description_fr": "Description suffisamment longue pour passer le filtre.",
            "longdescription_fr": "Autre description suffisamment longue pour l'embedding.",
            "firstdate_begin": "2025-06-01T10:00:00+00:00",
            "firstdate_end": "2025-06-01T12:00:00+00:00",
            "location_city": "Toulouse",
            "location_region": "Occitanie",
        }
    ]

    df = preprocess_events(events)

    assert isinstance(df.iloc[0]["begin"], pd.Timestamp)