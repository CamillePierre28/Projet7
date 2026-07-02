# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe argparse pour permettre de passer des options au script.
import argparse
# Importe json pour sauvegarder un rapport exploitable par une machine.
import json
# Importe subprocess pour lancer pytest depuis ce script.
import subprocess
# Importe sys pour utiliser le meme interpreteur Python que celui du projet.
import sys
# Importe datetime pour dater le rapport genere.
from datetime import datetime
# Importe Path pour manipuler les chemins de fichiers proprement.
from pathlib import Path
# Importe ElementTree pour lire le rapport XML genere par pytest.
from xml.etree import ElementTree as ET


# Dossier contenant les tests du projet.
TESTS_DIR = Path("tests")
# Dossier ou seront enregistres les rapports lisibles.
DOCS_DIR = Path("docs")
# Dossier ou seront enregistrees les donnees techniques de test.
REPORTS_DIR = Path("data/evaluation")
# Chemin du rapport Markdown final.
MARKDOWN_REPORT_PATH = DOCS_DIR / "resultats_tests.md"
# Chemin du rapport JSON final.
JSON_REPORT_PATH = REPORTS_DIR / "pytest_results.json"
# Chemin du rapport XML temporaire genere par pytest.
JUNIT_XML_PATH = REPORTS_DIR / "pytest_results.xml"
# Chemin du fichier contenant la sortie console complete de pytest.
PYTEST_OUTPUT_PATH = REPORTS_DIR / "pytest_output.txt"


# Transforme les secondes en texte court et lisible.
def format_duration(seconds: float) -> str:
    # Retourne la duree avec trois chiffres apres la virgule.
    return f"{seconds:.3f}s"


# Nettoie un texte pour eviter les blocs trop longs dans le rapport Markdown.
def shorten_text(text: str, max_chars: int = 2000) -> str:
    # Supprime les espaces inutiles au debut et a la fin.
    text = text.strip()
    # Si le texte est deja assez court, on le retourne tel quel.
    if len(text) <= max_chars:
        return text
    # Sinon, on coupe le texte et on indique qu'il a ete tronque.
    return text[:max_chars] + "\n... sortie tronquee ..."


# Recupere tous les fichiers de test presents dans le dossier tests.
def discover_test_files() -> list[str]:
    # Cherche tous les fichiers Python qui commencent par test_ dans le dossier tests.
    return sorted(str(path).replace("\\", "/") for path in TESTS_DIR.glob("test_*.py"))


# Lance pytest et produit un fichier XML JUnit exploitable.
def run_pytest(extra_pytest_args: list[str]) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    # Cree les dossiers de sortie si necessaire.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare la commande pytest.
    command = [
        # Utilise le Python courant.
        sys.executable,
        # Lance pytest comme module Python.
        "-m",
        "pytest",
        # Dossier contenant les tests.
        str(TESTS_DIR),
        # Mode verbeux pour afficher chaque test.
        "-vv",
        # Traceback court pour garder un rapport lisible.
        "--tb=short",
        # Demande a pytest de generer un rapport XML JUnit.
        f"--junitxml={JUNIT_XML_PATH}",
        # Desactive les couleurs dans la sortie pour faciliter la lecture du fichier texte.
        "--color=no",
    ]

    # Ajoute les arguments optionnels fournis par l'utilisateur.
    command.extend(extra_pytest_args)

    # Execute pytest et capture stdout/stderr.
    result = subprocess.run(
        # Commande complete.
        command,
        # Capture la sortie standard et la sortie d'erreur.
        capture_output=True,
        # Convertit les sorties en texte.
        text=True,
        # Encode explicitement en UTF-8.
        encoding="utf-8",
        # Remplace les caracteres non lisibles au lieu de planter.
        errors="replace",
    )

    # Sauvegarde toute la sortie console de pytest dans un fichier texte.
    PYTEST_OUTPUT_PATH.write_text(
        # Regroupe la commande, stdout et stderr.
        "Commande lancee : " + " ".join(command) + "\n\n"
        + "===== STDOUT =====\n"
        + result.stdout
        + "\n===== STDERR =====\n"
        + result.stderr,
        # Encode le fichier en UTF-8.
        encoding="utf-8",
    )

    # Retourne le resultat complet de la commande et la commande executee.
    return result, command


# Analyse le XML JUnit produit par pytest pour extraire les resultats test par test.
def parse_junit_xml() -> dict:
    # Si le fichier XML n'existe pas, on retourne une structure vide.
    if not JUNIT_XML_PATH.exists():
        return {
            "summary": {
                "tests": 0,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "time": 0.0,
            },
            "tests": [],
        }

    # Charge le fichier XML.
    tree = ET.parse(JUNIT_XML_PATH)
    # Recupere la racine du document XML.
    root = tree.getroot()

    # Le format peut etre testsuite ou testsuites selon la version de pytest.
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))

    # Initialise le resume global.
    summary = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "time": 0.0,
    }
    # Liste qui contiendra le detail de chaque test.
    test_results = []

    # Parcourt chaque suite de tests du XML.
    for suite in suites:
        # Ajoute le nombre de tests de cette suite.
        summary["tests"] += int(suite.attrib.get("tests", 0))
        # Ajoute le nombre d'echecs de cette suite.
        summary["failures"] += int(suite.attrib.get("failures", 0))
        # Ajoute le nombre d'erreurs de cette suite.
        summary["errors"] += int(suite.attrib.get("errors", 0))
        # Ajoute le nombre de tests ignores de cette suite.
        summary["skipped"] += int(suite.attrib.get("skipped", 0))
        # Ajoute la duree de cette suite.
        summary["time"] += float(suite.attrib.get("time", 0.0))

        # Parcourt chaque test individuel.
        for case in suite.findall("testcase"):
            # Recupere les informations principales du test.
            classname = case.attrib.get("classname", "")
            name = case.attrib.get("name", "")
            duration = float(case.attrib.get("time", 0.0))

            # Statut par defaut : le test est passe.
            status = "passed"
            # Message detaille en cas d'echec, erreur ou skip.
            message = ""
            # Type de probleme eventuel.
            problem_type = ""

            # Cherche un echec dans le test.
            failure = case.find("failure")
            # Cherche une erreur technique dans le test.
            error = case.find("error")
            # Cherche un test ignore.
            skipped = case.find("skipped")

            # Si le test a echoue, on recupere le message d'echec.
            if failure is not None:
                status = "failed"
                problem_type = failure.attrib.get("type", "failure")
                message = failure.attrib.get("message", "") or (failure.text or "")
            # Si le test a rencontre une erreur, on recupere le message d'erreur.
            elif error is not None:
                status = "error"
                problem_type = error.attrib.get("type", "error")
                message = error.attrib.get("message", "") or (error.text or "")
            # Si le test a ete ignore, on recupere la raison.
            elif skipped is not None:
                status = "skipped"
                problem_type = skipped.attrib.get("type", "skipped")
                message = skipped.attrib.get("message", "") or (skipped.text or "")

            # Ajoute le resultat du test dans la liste finale.
            test_results.append(
                {
                    "file_or_class": classname,
                    "name": name,
                    "status": status,
                    "duration_seconds": duration,
                    "problem_type": problem_type,
                    "message": message.strip(),
                }
            )

    # Retourne le resume et le detail des tests.
    return {
        "summary": summary,
        "tests": test_results,
    }


# Cree une phrase simple qui explique le role probable d'un fichier de test.
def describe_test_file(file_name: str) -> str:
    # Dictionnaire d'explications adaptees aux fichiers du projet.
    descriptions = {
        "test_api.py": "Vérifie les endpoints FastAPI, les réponses JSON et la gestion des erreurs.",
        "test_chunking.py": "Vérifie la transformation des événements en documents et leur découpage en chunks.",
        "test_embeddings.py": "Vérifie la présence et la structure du fichier d'embeddings généré.",
        "test_evaluation.py": "Vérifie les fonctions de scoring, de similarité et de règles métier.",
        "test_fetch_openagenda.py": "Vérifie la récupération des événements depuis OpenAgenda et certains champs attendus.",
        "test_preprocessing.py": "Vérifie le nettoyage HTML, la normalisation et la conversion des dates.",
        "test_vectorstore.py": "Vérifie la création, la sauvegarde, le chargement et la recherche dans FAISS.",
    }
    # Retourne la description connue ou une description generique.
    return descriptions.get(file_name, "Vérifie une partie du comportement du projet.")


# Genere le rapport Markdown lisible par un evaluateur.
def build_markdown_report(report: dict, pytest_result: subprocess.CompletedProcess[str], command: list[str]) -> str:
    # Recupere le resume global.
    summary = report["summary"]
    # Recupere le detail de chaque test.
    tests = report["tests"]
    # Recupere la date de generation.
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Recupere la liste des fichiers de test.
    test_files = discover_test_files()

    # Calcule le nombre de tests passes.
    passed = sum(1 for test in tests if test["status"] == "passed")
    # Calcule le nombre de tests echoues.
    failed = sum(1 for test in tests if test["status"] == "failed")
    # Calcule le nombre de tests en erreur.
    errors = sum(1 for test in tests if test["status"] == "error")
    # Calcule le nombre de tests ignores.
    skipped = sum(1 for test in tests if test["status"] == "skipped")

    # Commence le rapport Markdown.
    lines = [
        "# Résultats Des Tests",
        "",
        f"Rapport généré automatiquement le `{generated_at}`.",
        "",
        "## Objectif",
        "",
        "Ce document synthétise les résultats des tests présents dans le dossier `tests/`. ",
        "Il est généré par le script `scripts/generate_test_report.py`.",
        "",
        "## Commande Lancée",
        "",
        "```bash",
        " ".join(command),
        "```",
        "",
        "## Synthèse Globale",
        "",
        f"- Code retour pytest : `{pytest_result.returncode}`",
        f"- Tests détectés : `{summary['tests']}`",
        f"- Tests réussis : `{passed}`",
        f"- Tests échoués : `{failed}`",
        f"- Erreurs techniques : `{errors}`",
        f"- Tests ignorés : `{skipped}`",
        f"- Durée totale : `{format_duration(summary['time'])}`",
        "",
        "## Fichiers De Tests Pris En Compte",
        "",
    ]

    # Ajoute chaque fichier de test avec son role.
    for file_path in test_files:
        # Recupere seulement le nom du fichier.
        file_name = Path(file_path).name
        # Ajoute une ligne descriptive.
        lines.append(f"- `{file_path}` : {describe_test_file(file_name)}")

    # Ajoute la section detaillee.
    lines.extend(
        [
            "",
            "## Détail Technique Par Test",
            "",
            "| Test | Statut | Durée | Synthèse technique |",
            "| --- | --- | ---: | --- |",
        ]
    )

    # Ajoute une ligne par test.
    for test in tests:
        # Construit le nom complet du test.
        full_name = f"{test['file_or_class']}::{test['name']}"
        # Convertit le statut en libelle lisible.
        status = test["status"]
        # Prepare une synthese courte.
        if status == "passed":
            detail = "Le comportement attendu est validé."
        elif status == "skipped":
            detail = "Le test a été ignoré."
        else:
            detail = shorten_text(test["message"], max_chars=180).replace("\n", " ")
            if not detail:
                detail = "Le test n'a pas validé le comportement attendu."
        # Ajoute la ligne du tableau.
        lines.append(
            f"| `{full_name}` | `{status}` | `{format_duration(test['duration_seconds'])}` | {detail} |"
        )

    # Ajoute les sections de sortie console.
    lines.extend(
        [
            "",
            "## Sortie Console Pytest",
            "",
            "La sortie complète est enregistrée dans :",
            "",
            f"```text\n{PYTEST_OUTPUT_PATH.as_posix()}\n```",
            "",
            "Extrait de la sortie standard :",
            "",
            "```text",
            shorten_text(pytest_result.stdout, max_chars=3000),
            "```",
        ]
    )

    # Ajoute stderr seulement si elle existe.
    if pytest_result.stderr.strip():
        lines.extend(
            [
                "",
                "Extrait de la sortie d'erreur :",
                "",
                "```text",
                shorten_text(pytest_result.stderr, max_chars=2000),
                "```",
            ]
        )

    # Ajoute une conclusion automatique.
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
        ]
    )

    # Si tout passe, la conclusion est positive.
    if pytest_result.returncode == 0:
        lines.append("Tous les tests exécutés sont passés avec succès.")
    # Sinon, on explique qu'il faut regarder les details.
    else:
        lines.append(
            "Certains tests ont échoué ou rencontré une erreur. "
            "Les détails ci-dessus permettent d'identifier les points à corriger."
        )

    # Retourne le rapport Markdown complet.
    return "\n".join(lines) + "\n"


# Sauvegarde les resultats au format JSON et Markdown.
def save_reports(report: dict, pytest_result: subprocess.CompletedProcess[str], command: list[str]) -> None:
    # Ajoute des metadonnees utiles au rapport JSON.
    json_payload = {
        # Date de generation du rapport.
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        # Code retour de pytest.
        "pytest_returncode": pytest_result.returncode,
        # Fichiers de tests detectes.
        "test_files": discover_test_files(),
        # Resume et details extraits du XML.
        **report,
        # Chemin de la sortie console complete.
        "pytest_output_path": str(PYTEST_OUTPUT_PATH),
        # Chemin du rapport XML JUnit.
        "junit_xml_path": str(JUNIT_XML_PATH),
    }

    # Sauvegarde le JSON avec indentation pour qu'il reste lisible.
    JSON_REPORT_PATH.write_text(
        # Convertit le dictionnaire en JSON.
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        # Encode le fichier en UTF-8.
        encoding="utf-8",
    )

    # Construit le rapport Markdown.
    markdown = build_markdown_report(report, pytest_result, command)
    # Sauvegarde le rapport Markdown.
    MARKDOWN_REPORT_PATH.write_text(markdown, encoding="utf-8")


# Point d'entree principal du script.
def main() -> int:
    # Configure les arguments possibles du script.
    parser = argparse.ArgumentParser(
        description="Lance les tests pytest et genere une synthese technique Markdown + JSON."
    )
    # Lit les arguments connus du script et garde le reste pour pytest.
    args, pytest_args = parser.parse_known_args()

    # Lance pytest avec les arguments supplementaires.
    pytest_result, command = run_pytest(pytest_args)
    # Analyse le XML JUnit genere.
    report = parse_junit_xml()
    # Sauvegarde les rapports Markdown et JSON.
    save_reports(report, pytest_result, command)

    # Affiche les chemins des fichiers generes.
    print(f"Rapport Markdown genere : {MARKDOWN_REPORT_PATH}")
    print(f"Rapport JSON genere : {JSON_REPORT_PATH}")
    print(f"Sortie console pytest : {PYTEST_OUTPUT_PATH}")
    print(f"Rapport XML pytest : {JUNIT_XML_PATH}")

    # Retourne le meme code que pytest pour permettre l'utilisation en CI.
    return pytest_result.returncode


# Verifie que le fichier est lance directement.
if __name__ == "__main__":
    # Lance main et utilise son code retour comme code de sortie du script.
    raise SystemExit(main())