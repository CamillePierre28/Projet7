# Importe les fonctions d'evaluation utilisees dans les tests.
from scripts.evaluate_rag import (
    # Compare deux reponses pour voir si elles sont identiques.
    exact_match,
    # Transforme un score de similarite en classe lisible.
    classify_similarity,
    # Classe une reponse selon des regles metier simples.
    classify_business_rules,
)

# Importe la fonction qui verifie si une date est future.
from scripts.evaluate_rag import is_future_date


# Teste le cas ou deux reponses sont identiques en ignorant la casse.
def test_exact_match_positive():
    # Bonjour et bonjour doivent etre consideres comme identiques.
    assert exact_match("Bonjour", "bonjour") == 1


# Teste le cas ou deux reponses sont differentes.
def test_exact_match_negative():
    # Bonjour et Bonsoir ne doivent pas etre consideres comme identiques.
    assert exact_match("Bonjour", "Bonsoir") == 0


# Teste la classification d'un bon score de similarite.
def test_classify_similarity_correct():
    # Un score de 0.8 doit etre classe comme correct.
    assert classify_similarity(0.8) == "correcte"


# Teste la classification d'un score moyen.
def test_classify_similarity_partial():
    # Un score de 0.5 doit etre classe comme partiellement correct.
    assert classify_similarity(0.5) == "partiellement correcte"


# Teste la classification d'un mauvais score.
def test_classify_similarity_incorrect():
    # Un score de 0.1 doit etre classe comme incorrect.
    assert classify_similarity(0.1) == "incorrecte"


# Teste que les regles metier classent une bonne reponse comme correcte.
def test_business_classification_correct():
    # Calcule la classification avec toutes les regles au vert.
    result = classify_business_rules(
        # Nombre de sources trouvees.
        sources_count=5,
        # La ville attendue est presente.
        city_ok=True,
        # Le mot-cle attendu est present.
        keyword_ok=True,
        # Les dates sont futures quand c'est demande.
        future_ok=True,
        # La question contient une contrainte de ville.
        has_city_constraint=True,
        # La question contient une contrainte de mot-cle.
        has_keyword_constraint=True,
        # La question demande des evenements futurs.
        requires_future=True,
    )

    # Verifie que la classification finale est correcte.
    assert result == "correcte"


# Teste qu'une date tres lointaine est reconnue comme future.
def test_is_future_date():
    # L'annee 2099 doit etre dans le futur.
    assert is_future_date("2099-01-01T10:00:00+00:00") is True


# Teste qu'une date ancienne est reconnue comme passee.
def test_is_past_date():
    # L'annee 2020 doit etre dans le passe.
    assert is_future_date("2020-01-01T10:00:00+00:00") is False