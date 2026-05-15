# Puls-Events RAG Chatbot PoC

PoC de chatbot intelligent pour répondre à des questions sur des événements culturels à partir de données Open Agenda.

## Objectifs

* Récupérer des événements culturels depuis l’API Open Agenda.
* Filtrer les événements par zone géographique et période.
* Construire un index vectoriel avec FAISS.
* Orchestrer un système RAG avec LangChain.
* Générer des réponses en langage naturel avec Mistral.
* Exposer le système via une API REST FastAPI.
* Fournir des tests unitaires et un jeu de test annoté.
* Conteneuriser l’API pour une démonstration locale.

## Structure du projet

```text
api/              API FastAPI
scripts/          Scripts de collecte, preprocessing et indexation
src/              Logique métier RAG
tests/            Tests unitaires et fonctionnels
docs/             Documentation technique
data/raw/         Données brutes non versionnées
data/processed/   Données nettoyées non versionnées
vectorstore/      Index FAISS non versionné
```

## Installation

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Sur Windows :

```powershell
python -m venv env
env\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Variables d’environnement

Créer un fichier `.env` à partir du modèle :

```bash
cp .env.example .env
```

Puis renseigner :

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
OPENAGENDA_API_KEY=optional_if_needed
RAG_INDEX_PATH=vectorstore/faiss_index
DEFAULT_CITY=Paris
```

## Vérification de l’environnement

```bash
python check_env.py
```

## Remarques

Les dossiers `env/`, `.env`, `data/` et `vectorstore/` ne doivent pas être versionnés.

L’index vectoriel devra être reconstruit avec un script dédié afin de garantir la reproductibilité du projet sur une nouvelle machine.
