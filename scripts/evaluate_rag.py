# Permet d'utiliser les annotations de types sans les évaluer immédiatement
from __future__ import annotations

# Import des classes permettant de manipuler les dates et fuseaux horaires
from datetime import datetime, timezone

# Bibliothèque de manipulation de données sous forme de DataFrame
import pandas as pd

# Permet de transformer du texte en vecteurs TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Permet de calculer la similarité cosinus entre deux vecteurs
from sklearn.metrics.pairwise import cosine_similarity

# Import de la classe principale du système RAG
from src.rag import EventRAG

from src.deepeval_mistral import DeepEvalMistralLLM

from time import sleep

# Tentative d'import des métriques DeepEval
try:
    # Mesure la pertinence de la réponse
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRelevancyMetric,
    )

    # Classe représentant un cas de test pour DeepEval
    from deepeval.test_case import LLMTestCase

    # Indique que DeepEval est disponible
    DEEPEVAL_AVAILABLE = True

# Si DeepEval n'est pas installé
except Exception:
    DEEPEVAL_AVAILABLE = False


# Chemin vers le jeu de questions d'évaluation
TESTSET_PATH = "data/evaluation/test_questions.csv"

# Chemin du fichier CSV qui contiendra les résultats
OUTPUT_PATH = "data/evaluation/evaluation_results.csv"


def exact_match(prediction: str, reference: str) -> int:
    """
    Vérifie si la réponse générée est exactement identique à la réponse attendue.
    """
    # Suppression des espaces et comparaison sans tenir compte des majuscules
    return int(prediction.strip().lower() == reference.strip().lower())


def similarity_score(prediction: str, reference: str) -> float:
    """
    Calcule une similarité textuelle entre deux réponses grâce au TF-IDF.
    """

    # Création du vectoriseur
    vectorizer = TfidfVectorizer()

    # Transformation des deux textes en vecteurs numériques
    vectors = vectorizer.fit_transform([prediction, reference])

    # Calcul de la similarité cosinus entre les deux vecteurs
    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def classify_similarity(score: float) -> str:
    """
    Convertit un score numérique en une classe qualitative.
    """

    # Très proche de la référence
    if score >= 0.65:
        return "correcte"

    # Réponse partiellement correcte
    if score >= 0.35:
        return "partiellement correcte"

    # Réponse jugée incorrecte
    return "incorrecte"


def is_future_date(date_value: str) -> bool:
    """
    Vérifie si une date est située dans le futur.
    """

    try:
        # Conversion de la chaîne ISO en objet datetime
        event_date = datetime.fromisoformat(str(date_value).replace("Z", "+00:00"))

        # Compare avec la date actuelle UTC
        return event_date >= datetime.now(timezone.utc)

    # Si la date est invalide
    except ValueError:
        return False


def classify_business_rules(
    sources_count: int,
    city_ok: bool,
    keyword_ok: bool,
    future_ok: bool,
    has_city_constraint: bool,
    has_keyword_constraint: bool,
    requires_future: bool,
) -> str:
    """
    Évalue la réponse selon des règles métier.
    """

    # Première règle : au moins une source doit avoir été retrouvée
    checks = [sources_count > 0]

    # Vérification de la ville si une ville est attendue
    if has_city_constraint:
        checks.append(city_ok)

    # Vérification du mot-clé si nécessaire
    if has_keyword_constraint:
        checks.append(keyword_ok)

    # Vérification que les événements sont futurs si demandé
    if requires_future:
        checks.append(future_ok)

    # Calcul du pourcentage de règles respectées
    score = sum(checks) / len(checks)

    # Classification finale
    if score >= 0.8:
        return "correcte"

    if score >= 0.5:
        return "partiellement correcte"

    return "incorrecte"


def run_deepeval_metrics(
    question: str,
    generated_answer: str,
    retrieval_context: list[str],
) -> dict:
    """
    Lance les métriques DeepEval si elles sont disponibles.
    """

    # Si DeepEval n'est pas installé
    if not DEEPEVAL_AVAILABLE:
        return {
            "deepeval_available": False,
            "answer_relevancy_score": None,
            "faithfulness_score": None,
            "contextual_relevancy_score": None,
            "deepeval_error": "DeepEval non disponible",
        }

    try:

        # Création d'un cas de test DeepEval
        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            retrieval_context=retrieval_context,
        )

        # Instanciation des différentes métriques
        judge_model = DeepEvalMistralLLM(
            model_name="mistral-small-latest"
        )

        answer_relevancy = AnswerRelevancyMetric(
            model=judge_model,
        )

        faithfulness = FaithfulnessMetric(
            model=judge_model,
        )

        contextual_relevancy = ContextualRelevancyMetric(
            model=judge_model,
        )

        # Calcul de chaque métrique
        answer_relevancy.measure(test_case)
        sleep(2)

        faithfulness.measure(test_case)
        sleep(2)

        contextual_relevancy.measure(test_case)
        sleep(2)

        # Retour des scores obtenus
        return {
            "deepeval_available": True,
            "answer_relevancy_score": answer_relevancy.score,
            "faithfulness_score": faithfulness.score,
            "contextual_relevancy_score": contextual_relevancy.score,
            "deepeval_error": "",
        }

    # Gestion d'une éventuelle erreur d'exécution
    except Exception as exc:
        return {
            "deepeval_available": True,
            "answer_relevancy_score": None,
            "faithfulness_score": None,
            "contextual_relevancy_score": None,
            "deepeval_error": str(exc),
        }


def main() -> None:
    """
    Fonction principale d'évaluation.
    """

    # Chargement du système RAG
    rag = EventRAG()

    # Lecture du jeu de test
    df = pd.read_csv(TESTSET_PATH).fillna("")

    # Liste qui stockera les résultats
    rows = []

    # Parcours de chaque question du jeu de test
    for _, row in df.iterrows():

        # Lecture de la question
        question = row["question"]

        # Réponse de référence
        reference_answer = row.get("reference_answer", "")

        # Ville attendue
        expected_city = row.get("expected_city", "")

        # Mot-clé attendu
        expected_keyword = row.get("expected_keyword", "")

        # Indique si l'événement doit être futur
        requires_future = str(row.get("requires_future", "")).lower() == "true"

        # Appel du système RAG
        result = rag.answer(question)

        # Réponse générée
        generated_answer = result["answer"]

        # Sources utilisées par le RAG
        sources = result["sources"]

        # Construction du contexte envoyé à DeepEval
        retrieval_context = [
            f"{source.get('title', '')} "
            f"{source.get('city', '')} "
            f"{source.get('begin', '')} "
            f"{source.get('venue', '')} "
            f"{source.get('url', '')}"
            for source in sources
        ]

        # Si une réponse de référence existe
        if reference_answer:

            # Calcul du score de similarité
            sim_score = similarity_score(generated_answer, reference_answer)

            # Calcul de l'exact match
            em_score = exact_match(generated_answer, reference_answer)

            # Classe qualitative
            similarity_classification = classify_similarity(sim_score)

        # Sinon aucune comparaison possible
        else:
            sim_score = None
            em_score = None
            similarity_classification = "non évalué"

        # Concatène toutes les sources
        sources_text = " ".join(retrieval_context).lower()

        # Concatène réponse + sources pour les vérifications métier
        combined_text = f"{generated_answer.lower()} {sources_text}"

        # Vérifie si une ville est attendue
        has_city_constraint = bool(expected_city)

        # Vérifie si un mot-clé est attendu
        has_keyword_constraint = bool(expected_keyword)

        # Contrôle que la ville attendue apparaît
        city_ok = (
            expected_city.lower() in combined_text
            if has_city_constraint
            else True
        )

        # Contrôle que le mot-clé apparaît
        keyword_ok = (
            expected_keyword.lower() in combined_text
            if has_keyword_constraint
            else True
        )

        # Vérifie que toutes les dates des événements sont futures
        future_ok = (
            all(is_future_date(source.get("begin", "")) for source in sources)
            if requires_future and sources
            else True
        )

        # Évaluation métier globale
        business_classification = classify_business_rules(
            sources_count=len(sources),
            city_ok=city_ok,
            keyword_ok=keyword_ok,
            future_ok=future_ok,
            has_city_constraint=has_city_constraint,
            has_keyword_constraint=has_keyword_constraint,
            requires_future=requires_future,
        )

        # Calcul des métriques DeepEval
        deepeval_results = run_deepeval_metrics(
            question=question,
            generated_answer=generated_answer,
            retrieval_context=retrieval_context,
        )

        # Ajout des résultats de cette question
        rows.append(
            {
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "similarity_score": sim_score,
                "exact_match": em_score,
                "similarity_classification": similarity_classification,
                "sources_count": len(sources),
                "expected_city": expected_city,
                "city_ok": city_ok,
                "expected_keyword": expected_keyword,
                "keyword_ok": keyword_ok,
                "requires_future": requires_future,
                "future_ok": future_ok,
                "business_classification": business_classification,
                **deepeval_results,
            }
        )

    # Création du DataFrame final contenant toutes les évaluations
    results_df = pd.DataFrame(rows)

    # Sauvegarde des résultats dans un fichier CSV
    results_df.to_csv(OUTPUT_PATH, index=False)

    # Affichage d'un résumé des principaux indicateurs
    print(
        results_df[
            [
                "question",
                "similarity_score",
                "exact_match",
                "similarity_classification",
                "sources_count",
                "city_ok",
                "keyword_ok",
                "future_ok",
                "business_classification",
                "deepeval_available",
                "answer_relevancy_score",
                "faithfulness_score",
                "contextual_relevancy_score",
            ]
        ]
    )

    # Indique où le fichier a été enregistré
    print(f"Résultats sauvegardés dans : {OUTPUT_PATH}")


# Point d'entrée du programme
if __name__ == "__main__":

    # Lance la fonction principale
    main()