# Résultats Des Tests

## Objectif

Ce document synthétise les résultats des tests présents dans le dossier `tests/`. 
Il est généré par le script `scripts/generate_test_report.py`.

## Commande Lancée

```bash
C:\Users\cpier\Desktop\PROJET\puls-events-rag-chatbot\env\Scripts\python.exe -m pytest tests -vv --tb=short --junitxml=data\evaluation\pytest_results.xml --color=no
```

## Synthèse Globale

- Code retour pytest : `1`
- Tests détectés : `28`
- Tests réussis : `27`
- Tests échoués : `1`
- Erreurs techniques : `0`
- Tests ignorés : `0`
- Durée totale : `10.810s`

## Fichiers De Tests Pris En Compte

- `tests/test_api.py` : Vérifie les endpoints FastAPI, les réponses JSON et la gestion des erreurs.
- `tests/test_chunking.py` : Vérifie la transformation des événements en documents et leur découpage en chunks.
- `tests/test_embeddings.py` : Vérifie la présence et la structure du fichier d'embeddings généré.
- `tests/test_evaluation.py` : Vérifie les fonctions de scoring, de similarité et de règles métier.
- `tests/test_fetch_openagenda.py` : Vérifie la récupération des événements depuis OpenAgenda et certains champs attendus.
- `tests/test_preprocessing.py` : Vérifie le nettoyage HTML, la normalisation et la conversion des dates.
- `tests/test_vectorstore.py` : Vérifie la création, la sauvegarde, le chargement et la recherche dans FAISS.

## Détail Technique Par Test

| Test | Statut | Durée | Synthèse technique |
| --- | --- | ---: | --- |
| `tests.test_api::test_health_endpoint` | `passed` | `0.006s` | Le comportement attendu est validé. |
| `tests.test_api::test_ask_valid_question` | `passed` | `0.004s` | Le comportement attendu est validé. |
| `tests.test_api::test_ask_empty_question` | `passed` | `0.003s` | Le comportement attendu est validé. |
| `tests.test_api::test_ask_spaces_only_question` | `passed` | `0.003s` | Le comportement attendu est validé. |
| `tests.test_api::test_ask_missing_question` | `passed` | `0.002s` | Le comportement attendu est validé. |
| `tests.test_api::test_ask_empty_payload` | `passed` | `0.003s` | Le comportement attendu est validé. |
| `tests.test_api::test_rebuild_without_key_is_unauthorized` | `passed` | `0.004s` | Le comportement attendu est validé. |
| `tests.test_chunking::test_dataframe_to_documents_creates_documents` | `passed` | `0.002s` | Le comportement attendu est validé. |
| `tests.test_chunking::test_split_documents_creates_chunks` | `passed` | `0.001s` | Le comportement attendu est validé. |
| `tests.test_embeddings::test_embeddings_file_exists` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_embeddings::test_embeddings_have_expected_structure` | `passed` | `2.176s` | Le comportement attendu est validé. |
| `tests.test_evaluation::test_exact_match_positive` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_evaluation::test_exact_match_negative` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_evaluation::test_classify_similarity_correct` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_evaluation::test_classify_similarity_partial` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_evaluation::test_classify_similarity_incorrect` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_evaluation::test_business_classification_correct` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_evaluation::test_is_future_date` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_evaluation::test_is_past_date` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_fetch_openagenda::test_fetch_events_returns_results` | `passed` | `0.711s` | Le comportement attendu est validé. |
| `tests.test_fetch_openagenda::test_fetched_events_have_expected_fields` | `passed` | `0.542s` | Le comportement attendu est validé. |
| `tests.test_fetch_openagenda::test_fetched_events_are_filtered_on_occitanie_and_2025` | `failed` | `0.533s` | AssertionError: assert False  +  where False = <built-in method startswith of str object at 0x0000029951F10080>('2025')  +    where <built-in method startswith of str object at 0x0 ... sortie tronquee ... |
| `tests.test_preprocessing::test_clean_html_removes_tags` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_preprocessing::test_normalize_event_returns_expected_fields` | `passed` | `0.000s` | Le comportement attendu est validé. |
| `tests.test_preprocessing::test_preprocess_events_drops_invalid_events` | `passed` | `0.009s` | Le comportement attendu est validé. |
| `tests.test_preprocessing::test_preprocess_events_converts_dates` | `passed` | `0.005s` | Le comportement attendu est validé. |
| `tests.test_vectorstore::test_faiss_index_creation_and_search` | `passed` | `0.023s` | Le comportement attendu est validé. |
| `tests.test_vectorstore::test_faiss_save_and_load` | `passed` | `0.039s` | Le comportement attendu est validé. |

## Sortie Console Pytest

La sortie complète est enregistrée dans :

```text
data/evaluation/pytest_output.txt
```

Extrait de la sortie standard :

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\cpier\Desktop\PROJET\puls-events-rag-chatbot\env\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\cpier\Desktop\PROJET\puls-events-rag-chatbot
plugins: anyio-4.13.0, deepeval-4.0.6, langsmith-0.8.4, asyncio-1.4.0, cov-7.1.0, repeat-0.9.4, rerunfailures-16.3, xdist-3.8.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 28 items

tests/test_api.py::test_health_endpoint PASSED                           [  3%]
tests/test_api.py::test_ask_valid_question PASSED                        [  7%]
tests/test_api.py::test_ask_empty_question PASSED                        [ 10%]
tests/test_api.py::test_ask_spaces_only_question PASSED                  [ 14%]
tests/test_api.py::test_ask_missing_question PASSED                      [ 17%]
tests/test_api.py::test_ask_empty_payload PASSED                         [ 21%]
tests/test_api.py::test_rebuild_without_key_is_unauthorized PASSED       [ 25%]
tests/test_chunking.py::test_dataframe_to_documents_creates_documents PASSED [ 28%]
tests/test_chunking.py::test_split_documents_creates_chunks PASSED       [ 32%]
tests/test_embeddings.py::test_embeddings_file_exists PASSED             [ 35%]
tests/test_embeddings.py::test_embeddings_have_expected_structure PASSED [ 39%]
tests/test_evaluation.py::test_exact_match_positive PASSED               [ 42%]
tests/test_evaluation.py::test_exact_match_negative PASSED               [ 46%]
tests/test_evaluation.py::test_classify_similarity_correct PASSED        [ 50%]
tests/test_evaluation.py::test_classify_similarity_partial PASSED        [ 53%]
tests/test_evaluation.py::test_classify_similarity_incorrect PASSED      [ 57%]
tests/test_evaluation.py::test_business_classification_correct PASSED    [ 60%]
tests/test_evaluation.py::test_is_future_date PASSED                     [ 64%]
tests/test_evaluation.py::test_is_past_date PASSED                       [ 67%]
tests/test_fetch_openagenda.py::test_fetch_events_returns_results PASSED [ 71%]
tests/test_fetch_openagenda.py::test_fetched_events_have_expected_fields PASSED [ 75%]
tests/test_fetch_openagenda.py::test_fetched_events_are_filtered_on_occitanie_and_2025 FAILED [ 78%]
tests/test_preprocessing.py::test_clean_html_removes_tags PASSED         [ 82%]
tests/test_preprocessing.py::test_normalize_event_returns_expected_fields PASSED [ 85%]
tests/test_preprocessing.py::test_preprocess_events_drops_invalid_events PASSED [ 89%]
tests/test_preprocessing.py::test_preprocess_events_converts_dates PASSED [ 92%]
tests/test_vectorstore.py::test_faiss_index_creation_and_search PASSED   [ 96%]
tests/test_vectorstore.py::test_faiss_save_and_load PASSED               [100%]Running teardown with pytest sessionfinish...


================================== FAILURES ============================
... sortie tronquee ...
```

## Conclusion

Certains tests ont échoué ou rencontré une erreur. Les détails ci-dessus permettent d'identifier les points à corriger.

Le test échoué est : tests/test_fetch_openagenda.py::test_fetched_events_are_filtered_on_occitanie_and_2025. Pourquoi ? Le test attend une date qui commence par 2025, mais l’API a renvoyé un événement daté de 2032-01-01T01:00:00+00:00.
Ce test est trop strict : il ne devrait pas imposer uniquement 2025 si ton API peut renvoyer des événements futurs. Il vaudrait mieux vérifier que la région est Occitanie et que la date existe/est valide.