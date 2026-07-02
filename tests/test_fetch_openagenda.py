# Importe la fonction qui recupere les evenements depuis OpenAgenda.
from scripts.fetch_openagenda import fetch_events


# Teste que l'appel a l'API retourne bien une liste d'evenements.
def test_fetch_events_returns_results():
    # Recupere une petite page de 10 evenements maximum.
    events = fetch_events(limit=10, max_pages=1)

    # Verifie que le resultat est bien une liste Python.
    assert isinstance(events, list)
    # Verifie que la liste contient au moins un evenement.
    assert len(events) > 0


# Teste que les evenements recuperes contiennent les champs attendus.
def test_fetched_events_have_expected_fields():
    # Recupere une petite page d'evenements depuis l'API.
    events = fetch_events(limit=10, max_pages=1)
    # Prend le premier evenement pour verifier sa structure.
    event = events[0]

    # Liste les champs indispensables attendus dans un evenement brut.
    expected_fields = [
        # Identifiant unique de l'evenement.
        "uid",
        # Titre en francais.
        "title_fr",
        # Description en francais.
        "description_fr",
        # Date de debut.
        "firstdate_begin",
        # Region de l'evenement.
        "location_region",
    ]

    # Parcourt chaque champ attendu.
    for field in expected_fields:
        # Verifie que le champ est bien present dans l'evenement.
        assert field in event


# Teste que les evenements recuperes sont bien situes en Occitanie et en 2025.
def test_fetched_events_are_filtered_on_occitanie_and_2025():
    # Recupere une petite page d'evenements depuis l'API.
    events = fetch_events(limit=20, max_pages=1)

    # Verifie que l'API a retourne au moins un evenement.
    assert len(events) > 0

    # Parcourt tous les evenements recuperes.
    for event in events:
        # Verifie que la region de chaque evenement est bien Occitanie.
        assert event["location_region"] == "Occitanie"
        # Verifie que la date de debut commence par 2025.
        assert str(event["firstdate_begin"]).startswith("2025")