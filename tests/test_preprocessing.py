# Importe pandas pour verifier les types de donnees retournes par le preprocessing.
import pandas as pd

# Importe les fonctions de preprocessing a tester.
from src.preprocessing import (
    # Nettoie le HTML et les espaces.
    clean_html,
    # Transforme un evenement brut en evenement normalise.
    normalize_event,
    # Nettoie une liste complete d'evenements.
    preprocess_events,
)


# Teste que les balises HTML sont bien supprimees.
def test_clean_html_removes_tags():
    # Cree un texte contenant des balises HTML.
    text = "<p>Concert <strong>gratuit</strong></p>"
    # Verifie que le texte nettoye ne contient plus les balises.
    assert clean_html(text) == "Concert gratuit"


# Teste que la normalisation retourne les champs attendus.
def test_normalize_event_returns_expected_fields():
    # Cree un faux evenement au format proche de l'API OpenAgenda.
    event = {
        # Identifiant unique de l'evenement.
        "uid": "123",
        # Titre en francais.
        "title_fr": "Exposition Monet",
        # Description courte.
        "description_fr": "Une belle exposition",
        # Description longue avec HTML.
        "longdescription_fr": "<p>Description détaillée</p>",
        # URL publique de l'evenement.
        "canonicalurl": "https://example.com/event",
        # Date de debut.
        "firstdate_begin": "2025-06-01T10:00:00+00:00",
        # Date de fin.
        "firstdate_end": "2025-06-01T18:00:00+00:00",
        # Nom du lieu.
        "location_name": "Musée Test",
        # Adresse du lieu.
        "location_address": "1 rue Exemple",
        # Ville.
        "location_city": "Toulouse",
        # Departement.
        "location_department": "Haute-Garonne",
        # Region.
        "location_region": "Occitanie",
    }

    # Normalise le faux evenement.
    normalized = normalize_event(event)

    # Verifie que l'identifiant est conserve.
    assert normalized["uid"] == "123"
    # Verifie que le titre est bien extrait.
    assert normalized["title"] == "Exposition Monet"
    # Verifie que la ville est bien extraite.
    assert normalized["city"] == "Toulouse"
    # Verifie que la region est bien extraite.
    assert normalized["region"] == "Occitanie"
    # Verifie que le titre est present dans le texte utilise pour les embeddings.
    assert "Exposition Monet" in normalized["text_for_embedding"]


# Teste que les evenements invalides sont supprimes.
def test_preprocess_events_drops_invalid_events():
    # Cree une liste avec un evenement valide et un evenement invalide.
    events = [
        # Evenement valide.
        {
            # Identifiant present.
            "uid": "1",
            # Titre present.
            "title_fr": "Concert valide",
            # Description assez longue pour passer le filtre.
            "description_fr": "Description suffisamment longue pour être indexée dans le futur système RAG.",
            # Description longue utile pour le texte d'embedding.
            "longdescription_fr": "Description longue également exploitable.",
            # Date de debut valide.
            "firstdate_begin": "2025-06-01T10:00:00+00:00",
            # Date de fin valide.
            "firstdate_end": "2025-06-01T12:00:00+00:00",
            # Ville de l'evenement.
            "location_city": "Toulouse",
            # Region de l'evenement.
            "location_region": "Occitanie",
        },
        # Evenement invalide car il n'a pas d'UID.
        {
            # Identifiant manquant.
            "uid": None,
            # Titre present mais insuffisant sans UID.
            "title_fr": "Sans UID",
            # Date de debut.
            "firstdate_begin": "2025-06-01T10:00:00+00:00",
            # Ville.
            "location_city": "Toulouse",
            # Region.
            "location_region": "Occitanie",
        },
    ]

    # Lance le preprocessing sur les deux evenements.
    df = preprocess_events(events)

    # Verifie qu'un seul evenement est conserve.
    assert len(df) == 1
    # Verifie que l'evenement conserve est celui avec l'UID 1.
    assert df.iloc[0]["uid"] == "1"


# Teste que les dates sont converties au format pandas Timestamp.
def test_preprocess_events_converts_dates():
    # Cree un evenement valide avec des dates ISO.
    events = [
        {
            # Identifiant unique.
            "uid": "1",
            # Titre de l'evenement.
            "title_fr": "Événement test",
            # Description assez longue pour passer le filtre.
            "description_fr": "Description suffisamment longue pour passer le filtre.",
            # Description longue assez riche pour l'embedding.
            "longdescription_fr": "Autre description suffisamment longue pour l'embedding.",
            # Date de debut au format ISO.
            "firstdate_begin": "2025-06-01T10:00:00+00:00",
            # Date de fin au format ISO.
            "firstdate_end": "2025-06-01T12:00:00+00:00",
            # Ville.
            "location_city": "Toulouse",
            # Region.
            "location_region": "Occitanie",
        }
    ]

    # Lance le preprocessing.
    df = preprocess_events(events)

    # Verifie que la colonne begin est bien convertie en Timestamp pandas.
    assert isinstance(df.iloc[0]["begin"], pd.Timestamp)