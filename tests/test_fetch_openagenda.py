from scripts.fetch_openagenda import fetch_events


def test_fetch_events_returns_results():
    events = fetch_events(limit=10, max_pages=1)

    assert isinstance(events, list)
    assert len(events) > 0


def test_fetched_events_have_expected_fields():
    events = fetch_events(limit=10, max_pages=1)
    event = events[0]

    expected_fields = [
        "uid",
        "title_fr",
        "description_fr",
        "firstdate_begin",
        "location_region",
    ]

    for field in expected_fields:
        assert field in event


def test_fetched_events_are_filtered_on_occitanie_and_2025():
    events = fetch_events(limit=20, max_pages=1)

    assert len(events) > 0

    for event in events:
        assert event["location_region"] == "Occitanie"
        assert str(event["firstdate_begin"]).startswith("2025")