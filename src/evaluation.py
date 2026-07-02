# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe datetime pour manipuler les dates et timezone pour travailler en UTC.
from datetime import datetime, timezone


# Verifie si une date donnee correspond a une date future.
def is_future_date(date_value: str) -> bool:
    # Tente de convertir le texte recu en vraie date Python.
    try:
        # Convertit une date ISO en datetime et remplace Z par +00:00 pour gerer l'UTC.
        event_date = datetime.fromisoformat(str(date_value).replace("Z", "+00:00"))
        # Compare la date de l'evenement avec la date actuelle en UTC.
        return event_date >= datetime.now(timezone.utc)
    # Si la date est invalide, la fonction retourne False.
    except ValueError:
        # Une date invalide n'est pas consideree comme future.
        return False


# Detecte si la question utilisateur demande des evenements futurs.
def requires_future_answer(question: str) -> bool:
    # Liste de mots qui indiquent souvent une demande d'evenements a venir.
    keywords = [
        # Forme masculine singuliere.
        "prévu",
        # Forme masculine plurielle.
        "prévus",
        # Forme feminine singuliere.
        "prévue",
        # Forme feminine plurielle.
        "prévues",
        # Expression directe pour parler du futur.
        "à venir",
        # Mot indiquant le prochain evenement.
        "prochain",
        # Forme feminine de prochain.
        "prochaine",
        # Forme plurielle masculine.
        "prochains",
        # Forme plurielle feminine.
        "prochaines",
        # Mot indiquant le jour suivant.
        "demain",
        # Mot indiquant le week-end.
        "week-end",
        # Variante sans tiret.
        "ce weekend",
        # Mot indiquant bientot.
        "bientôt",
    ]

    # Met la question en minuscules pour rendre la recherche insensible a la casse.
    question_lower = question.lower()
    # Retourne True si au moins un mot-cle est present dans la question.
    return any(keyword in question_lower for keyword in keywords)


# Essaie de deviner la ville attendue dans une question.
def infer_expected_city(question: str) -> str:
    # Liste simple de villes connues dans le perimetre du projet.
    known_cities = [
        # Ville connue.
        "Toulouse",
        # Ville connue.
        "Montpellier",
        # Ville connue.
        "Nîmes",
        # Ville connue.
        "Alès",
        # Ville connue.
        "Perpignan",
        # Ville connue.
        "Béziers",
        # Ville connue.
        "Carcassonne",
        # Ville connue.
        "Montauban",
        # Ville connue.
        "Tarbes",
        # Ville connue.
        "Narbonne",
    ]

    # Met la question en minuscules pour comparer plus facilement.
    question_lower = question.lower()

    # Parcourt les villes connues.
    for city in known_cities:
        # Si la ville apparait dans la question, on la retourne.
        if city.lower() in question_lower:
            # Retourne la ville detectee.
            return city

    # Retourne une chaine vide si aucune ville connue n'est detectee.
    return ""


# Essaie de deviner le type d'evenement attendu dans une question.
def infer_expected_keyword(question: str) -> str:
    # Liste simple de mots-cles culturels connus.
    known_keywords = [
        # Type d'evenement.
        "danse",
        # Type d'evenement.
        "concert",
        # Type d'evenement.
        "musique",
        # Type d'evenement.
        "exposition",
        # Type d'evenement.
        "théâtre",
        # Type d'evenement.
        "cinéma",
        # Type d'evenement.
        "festival",
        # Theme d'evenement.
        "tourisme",
        # Type d'evenement.
        "atelier",
        # Type d'evenement.
        "visite",
    ]

    # Met la question en minuscules pour comparer plus facilement.
    question_lower = question.lower()

    # Parcourt les mots-cles connus.
    for keyword in known_keywords:
        # Si le mot-cle apparait dans la question, on le retourne.
        if keyword in question_lower:
            # Retourne le mot-cle detecte.
            return keyword

    # Retourne une chaine vide si aucun mot-cle connu n'est detecte.
    return ""


# Classe une reponse selon des regles metier simples.
def classify_business_rules(
    sources_count: int,
    city_ok: bool,
    keyword_ok: bool,
    future_ok: bool,
    has_city_constraint: bool,
    has_keyword_constraint: bool,
    requires_future: bool,
) -> str:
    # La premiere regle exige d'avoir au moins une source.
    checks = [sources_count > 0]

    # Si la question demande une ville, on ajoute la verification de ville.
    if has_city_constraint:
        # Ajoute le resultat du controle de ville.
        checks.append(city_ok)

    # Si la question demande un mot-cle, on ajoute la verification de mot-cle.
    if has_keyword_constraint:
        # Ajoute le resultat du controle de mot-cle.
        checks.append(keyword_ok)

    # Si la question demande du futur, on ajoute la verification de date future.
    if requires_future:
        # Ajoute le resultat du controle temporel.
        checks.append(future_ok)

    # Calcule la proportion de controles reussis.
    score = sum(checks) / len(checks)

    # Si au moins 80 % des controles passent, la reponse est correcte.
    if score >= 0.8:
        # Retourne la meilleure classification.
        return "correcte"
    # Si au moins 50 % des controles passent, la reponse est partiellement correcte.
    if score >= 0.5:
        # Retourne une classification intermediaire.
        return "partiellement correcte"
    # Sinon, la reponse est consideree incorrecte.
    return "incorrecte"


# Evalue automatiquement une reponse juste apres un appel API.
def evaluate_answer_online(
    question: str,
    answer: str,
    sources: list[dict],
) -> dict:
    # Devine la ville attendue a partir de la question.
    expected_city = infer_expected_city(question)
    # Devine le mot-cle attendu a partir de la question.
    expected_keyword = infer_expected_keyword(question)
    # Detecte si la question demande des evenements futurs.
    future_required = requires_future_answer(question)

    # Regroupe le texte important de toutes les sources dans une seule chaine.
    sources_text = " ".join(
        [
            # Construit un petit texte avec les champs utiles d'une source.
            f"{source.get('title', '')} "
            f"{source.get('city', '')} "
            f"{source.get('begin', '')} "
            f"{source.get('venue', '')} "
            f"{source.get('url', '')}"
            # Repete ce format pour chaque source.
            for source in sources
        ]
    ).lower()

    # Combine la reponse et les sources pour faire les verifications textuelles.
    combined_text = f"{answer.lower()} {sources_text}"

    # Indique si une contrainte de ville a ete detectee.
    has_city_constraint = bool(expected_city)
    # Indique si une contrainte de mot-cle a ete detectee.
    has_keyword_constraint = bool(expected_keyword)

    # Verifie si la ville attendue apparait dans la reponse ou les sources.
    city_ok = (
        # Cherche la ville en minuscules dans le texte combine.
        expected_city.lower() in combined_text
        # Fait ce controle seulement si une ville est attendue.
        if has_city_constraint
        # Si aucune ville n'est attendue, la regle est automatiquement valide.
        else True
    )

    # Verifie si le mot-cle attendu apparait dans la reponse ou les sources.
    keyword_ok = (
        # Cherche le mot-cle en minuscules dans le texte combine.
        expected_keyword.lower() in combined_text
        # Fait ce controle seulement si un mot-cle est attendu.
        if has_keyword_constraint
        # Si aucun mot-cle n'est attendu, la regle est automatiquement valide.
        else True
    )

    # Verifie si toutes les sources sont futures quand la question l'exige.
    future_ok = (
        # Controle chaque date de debut des sources.
        all(is_future_date(source.get("begin", "")) for source in sources)
        # Fait ce controle seulement si le futur est demande et qu'il y a des sources.
        if future_required and sources
        # Sinon, la regle est consideree valide.
        else True
    )

    # Calcule la classification finale avec les regles metier.
    classification = classify_business_rules(
        # Nombre de sources utilisees.
        sources_count=len(sources),
        # Resultat du controle de ville.
        city_ok=city_ok,
        # Resultat du controle de mot-cle.
        keyword_ok=keyword_ok,
        # Resultat du controle de date future.
        future_ok=future_ok,
        # Indique si la ville etait une contrainte.
        has_city_constraint=has_city_constraint,
        # Indique si le mot-cle etait une contrainte.
        has_keyword_constraint=has_keyword_constraint,
        # Indique si le futur etait demande.
        requires_future=future_required,
    )

    # Retourne un dictionnaire qui sera renvoye par l'API.
    return {
        # Nombre de sources trouvees.
        "sources_count": len(sources),
        # Ville detectee dans la question.
        "expected_city": expected_city,
        # Resultat du controle de ville.
        "city_ok": city_ok,
        # Mot-cle detecte dans la question.
        "expected_keyword": expected_keyword,
        # Resultat du controle de mot-cle.
        "keyword_ok": keyword_ok,
        # Indique si la question demandait des evenements futurs.
        "requires_future": future_required,
        # Resultat du controle temporel.
        "future_ok": future_ok,
        # Classification finale de la reponse.
        "classification": classification,
    }