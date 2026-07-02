```markdown
# Puls-Events RAG Chatbot

Chatbot RAG développé en Python pour répondre à des questions sur des événements culturels à partir de données OpenAgenda / OpenDataSoft.

Le projet récupère des événements, les nettoie, les transforme en documents exploitables, les indexe dans une base vectorielle FAISS, puis utilise Mistral pour générer des réponses en langage naturel avec les sources utilisées.

## Objectif Du Projet

L'objectif est de construire un assistant capable de répondre à des questions comme :

```text
Quels événements culturels sont prévus à Toulouse ?
Y a-t-il des concerts à Montpellier ?
Quels festivals sont à venir en Occitanie ?
```

Le chatbot ne répond pas uniquement avec les connaissances générales du modèle. Il cherche d'abord les événements pertinents dans une base locale, puis demande à Mistral de rédiger une réponse à partir de ces événements.

Cette approche permet de :

- limiter les hallucinations 
- fournir des sources vérifiables 
- contrôler les données utilisées 
- mettre à jour le corpus en reconstruisant l'index 
- répondre à des questions formulées naturellement

## Principe Général

Le projet repose sur une architecture RAG, pour Retrieval-Augmented Generation.

En version simple :

```text
Données OpenAgenda
    ↓
Nettoyage des événements
    ↓
Création de documents texte
    ↓
Embeddings Mistral
    ↓
Index vectoriel FAISS
    ↓
Question utilisateur
    ↓
Recherche des événements pertinents
    ↓
Réponse générée par Mistral
```

Le rôle de chaque partie est séparé :

- OpenAgenda fournit les données d'événements.
- Pandas nettoie et structure les données.
- Mistral Embeddings transforme les textes en vecteurs numériques.
- FAISS retrouve les textes les plus proches du sens de la question.
- Mistral Chat génère la réponse finale.
- FastAPI expose le système sous forme d'API REST.

## Technologies Utilisées

| Technologie | Rôle dans le projet |
| --- | --- |
| Python | Langage principal du projet |
| FastAPI | Création de l'API REST |
| Uvicorn | Serveur qui lance l'API FastAPI |
| Pandas | Nettoyage, transformation et export des données |
| Requests | Appels HTTP vers l'API OpenAgenda |
| LangChain | Orchestration des documents, prompts, modèles et vectorstore |
| Mistral | Embeddings et génération de réponse |
| FAISS | Recherche vectorielle locale |
| Pydantic | Validation des entrées et sorties API |
| Pytest | Tests automatisés |
| Docker | Exécution reproductible du projet |

## Structure Du Projet

```text
api/
  main.py                    API FastAPI

src/
  rag.py                     Cœur du chatbot RAG
  preprocessing.py           Nettoyage et normalisation des événements
  chunking.py                Transformation des événements en documents et chunks
  mistral_embeddings.py      Wrapper d'embeddings Mistral compatible LangChain
  evaluation.py              Évaluation métier légère des réponses
  deepeval_mistral.py        Adaptateur Mistral pour DeepEval

scripts/
  fetch_openagenda.py        Récupération et préparation des événements
  build_vectorstore.py       Construction de l'index FAISS
  ask_rag.py                 Test manuel du RAG sans API
  api_test.py                Test manuel des endpoints API
  evaluate_rag.py            Évaluation sur un jeu de questions
  search_vectorstore.py      Recherche directe dans le vectorstore

data/
  raw/                       Données brutes récupérées depuis l'API
  processed/                 Données nettoyées
  evaluation/                Questions et résultats d'évaluation

vectorstore/
  faiss_index/               Index FAISS sauvegardé localement

tests/
  test_*.py                  Tests unitaires et fonctionnels
```

## Fonctionnement Détaillé

### 1. Récupération des événements

Le script `scripts/fetch_openagenda.py` récupère les événements depuis l'API OpenAgenda / OpenDataSoft.

Il utilise une pagination avec un `offset`, car l'API ne renvoie pas tous les événements en une seule réponse.

Exemple de logique :

```text
page 0 → événements 0 à 99
page 1 → événements 100 à 199
page 2 → événements 200 à 299
```

Les données sont sauvegardées en deux versions :

- `data/raw/events_raw.json` : données brutes telles que reçues depuis l'API 
- `data/processed/events_processed.csv` : données nettoyées au format CSV 
- `data/processed/events_processed.json` : données nettoyées au format JSON

La sauvegarde des données brutes permet de garder une trace exacte de la source et de faciliter le débogage.

### 2. Nettoyage et normalisation

Le fichier `src/preprocessing.py` transforme les événements bruts en données propres.

Les traitements effectués sont notamment :

- suppression des balises HTML 
- normalisation des espaces 
- extraction des champs utiles 
- suppression des événements sans identifiant 
- suppression des doublons 
- conversion des dates 
- suppression des événements sans titre ou sans date 
- suppression des textes trop courts 
- tri chronologique 
- conservation des événements récents et des événements à venir

Les champs techniques de l'API sont transformés en champs plus simples.

Exemple :

```text
title_fr              → title
description_fr        → description
longdescription_fr    → long_description
location_city         → city
location_name         → venue
firstdate_begin       → begin
canonicalurl          → url
```

Le code crée aussi un champ `text_for_embedding`. Ce champ regroupe les informations importantes d'un événement dans un seul texte :

```text
Titre : ...
Description : ...
Lieu : ...
Ville : ...
Département : ...
Région : ...
Date de début : ...
URL : ...
```

Ce texte sert ensuite à créer les embeddings.

### 3. Transformation en documents et chunks

Le fichier `src/chunking.py` transforme chaque événement en document LangChain.

Un document contient :

- un texte principal, utilisé pour la recherche sémantique
- des métadonnées, utilisées pour afficher les sources

Les métadonnées conservées sont par exemple :

```text
uid
title
begin
end
venue
address
city
department
region
url
```

Les documents sont ensuite découpés en morceaux appelés `chunks`.

Le découpage est utile parce qu'un texte long peut contenir plusieurs informations. Des chunks plus courts permettent une recherche plus précise.

Configuration actuelle :

```text
chunk_size = 800
chunk_overlap = 120
```

Le chevauchement de 120 caractères évite de perdre une information importante si elle est coupée entre deux morceaux.

### 4. Création des embeddings

Le fichier `src/mistral_embeddings.py` définit une classe `MistralEmbeddings`.

Un embedding est une représentation numérique du sens d'un texte. Concrètement, chaque texte est transformé en liste de nombres.

Deux textes proches en sens auront des vecteurs proches, même s'ils n'utilisent pas exactement les mêmes mots.

Exemple :

```text
concert à Toulouse
soirée musicale en Haute-Garonne
spectacle de jazz
```

Ces textes ne sont pas identiques, mais ils peuvent être proches sémantiquement.

Le modèle utilisé est :

```text
mistral-embed
```

Le code gère aussi les limites d'API. Si Mistral renvoie une erreur de rate limit, le script attend puis réessaie avec un temps d'attente progressif.

### 5. Construction de l'index FAISS

Le script `scripts/build_vectorstore.py` construit l'index FAISS.

Il effectue les étapes suivantes :

```text
1. Charger les événements nettoyés depuis data/processed/events_processed.csv
2. Transformer les lignes en documents LangChain
3. Découper les documents en chunks
4. Créer les embeddings avec Mistral
5. Stocker les vecteurs dans FAISS
6. Sauvegarder l'index dans vectorstore/faiss_index
```

FAISS permet de faire une recherche par similarité vectorielle. Contrairement à une recherche par mots-clés, il ne cherche pas seulement les mêmes mots : il cherche les textes les plus proches du sens de la question.

### 6. Réponse aux questions avec le RAG

Le fichier `src/rag.py` contient le cœur du chatbot.

La classe principale est :

```python
EventRAG
```

Au démarrage, cette classe :

- charge les variables d'environnement 
- initialise les embeddings Mistral 
- recharge l'index FAISS local 
- initialise le modèle de chat Mistral 
- prépare le prompt de réponse

Quand une question est posée, le fonctionnement est le suivant :

```text
Question utilisateur
    ↓
Recherche dans FAISS
    ↓
Récupération des documents les plus pertinents
    ↓
Filtrage éventuel des événements passés
    ↓
Construction d'un contexte texte
    ↓
Envoi du contexte et de la question à Mistral
    ↓
Réponse finale
    ↓
Retour des sources utilisées
```

Le prompt impose au modèle de répondre uniquement à partir du contexte fourni.

Cela permet de réduire le risque d'invention, car Mistral reçoit une consigne claire :

```text
Si le contexte ne contient pas l'information, il faut le dire.
Il ne faut jamais fabriquer d'événement.
```

### 7. Filtrage des événements futurs

Le code détecte si une question concerne des événements futurs.

Il cherche des mots-clés comme :

```text
prévu
prévus
à venir
prochain
prochaine
bientôt
demain
week-end
```

Si ces mots sont trouvés, le système filtre les documents récupérés et garde uniquement les événements dont la date est supérieure ou égale à la date actuelle.

Exemple :

```text
Quels événements sont prévus à Toulouse ?
```

Dans ce cas, le chatbot évite de répondre avec des événements déjà passés.

### 8. API FastAPI

Le fichier `api/main.py` expose le chatbot via une API REST.

L'API charge le RAG au démarrage. Cela évite de recharger FAISS et Mistral à chaque question.

Endpoints disponibles :

| Méthode | Route | Description |
| --- | --- | --- |
| GET | `/health` | Vérifie que l'API fonctionne et que le RAG est chargé |
| POST | `/ask` | Pose une question au chatbot |
| POST | `/rebuild` | Reconstruit les données et l'index FAISS |

### 9. Évaluation automatique

Le fichier `src/evaluation.py` ajoute une évaluation légère après chaque réponse.

L'évaluation vérifie :

- le nombre de sources utilisées 
- si la ville demandée apparaît dans la réponse ou les sources 
- si le mot-clé demandé apparaît 
- si les événements sont bien futurs lorsque la question l'exige

La réponse est classée en :

```text
correcte
partiellement correcte
incorrecte
```

Cette évaluation ne remplace pas une analyse humaine, mais elle donne un premier indicateur utile pour tester le système.

## Installation Locale

### 1. Créer un environnement virtuel

Sur Windows :

```powershell
python -m venv env
env\Scripts\Activate.ps1
```

Sur Linux ou macOS :

```bash
python -m venv env
source env/bin/activate
```

### 2. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

Créer un fichier `.env` à la racine du projet.

Exemple :

```env
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_CHAT_MODEL=mistral-small-latest
MISTRAL_EMBED_BATCH_SIZE=8
MISTRAL_EMBED_SLEEP_SECONDS=2

OPENAGENDA_API_URL=https://example.com/api/explore/v2.1/catalog/datasets/your-dataset/records
EVENT_MAX_PAGES=50
EVENT_LOOKBACK_DAYS=365

RAG_FETCH_K=20
RAG_TOP_K=5

REBUILD_API_KEY=your_rebuild_secret
```

Variables principales :

| Variable | Description |
| --- | --- |
| `MISTRAL_API_KEY` | Clé API Mistral obligatoire |
| `MISTRAL_CHAT_MODEL` | Modèle utilisé pour générer la réponse |
| `OPENAGENDA_API_URL` | URL de l'API source des événements |
| `EVENT_MAX_PAGES` | Nombre maximal de pages récupérées |
| `EVENT_LOOKBACK_DAYS` | Nombre de jours passés conservés |
| `RAG_FETCH_K` | Nombre de documents récupérés au départ |
| `RAG_TOP_K` | Nombre final de documents utilisés dans le contexte |
| `REBUILD_API_KEY` | Clé protégeant l'endpoint `/rebuild` |

### 4. Vérifier l'environnement

```bash
python check_env.py
```

Cette commande vérifie que les bibliothèques principales sont installées et que la clé Mistral est disponible.

## Utilisation Locale

### 1. Récupérer et préparer les événements

```bash
python -m scripts.fetch_openagenda
```

Cette commande crée ou met à jour :

```text
data/raw/events_raw.json
data/processed/events_processed.csv
data/processed/events_processed.json
```

### 2. Construire l'index FAISS

```bash
python -m scripts.build_vectorstore
```

Cette commande crée ou met à jour :

```text
vectorstore/faiss_index
```

### 3. Tester le RAG sans API

```bash
python -m scripts.ask_rag
```

Ce script pose une question prédéfinie au système RAG et affiche la réponse avec les sources.

### 4. Lancer l'API

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

L'API est ensuite disponible à l'adresse :

```text
http://127.0.0.1:8000
```

La documentation interactive FastAPI est disponible ici :

```text
http://127.0.0.1:8000/docs
```

## Exemples D'appels API

### Vérifier l'état de l'API

```bash
curl http://127.0.0.1:8000/health
```

Exemple de réponse :

```json
{
  "status": "ok",
  "rag_loaded": true
}
```

### Poser une question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Quels événements culturels sont prévus à Toulouse ?\"}"
```

Réponse attendue :

```json
{
  "question": "Quels événements culturels sont prévus à Toulouse ?",
  "answer": "...",
  "sources": [
    {
      "uid": "...",
      "title": "...",
      "city": "Toulouse",
      "begin": "...",
      "venue": "...",
      "url": "..."
    }
  ],
  "evaluation": {
    "sources_count": 5,
    "expected_city": "Toulouse",
    "city_ok": true,
    "expected_keyword": "",
    "keyword_ok": true,
    "requires_future": true,
    "future_ok": true,
    "classification": "correcte"
  }
}
```

### Reconstruire l'index depuis l'API

```bash
curl -X POST http://127.0.0.1:8000/rebuild \
  -H "X-API-Key: your_rebuild_secret"
```

Cette route relance :

```text
python -m scripts.fetch_openagenda
python -m scripts.build_vectorstore
```

Puis elle recharge le système RAG.

## Utilisation Avec Docker

Le projet peut être lancé avec Docker et Docker Compose.

### Construire et lancer les services

```bash
docker compose up --build
```

Le fichier `docker-compose.yml` définit deux services :

| Service | Rôle |
| --- | --- |
| `vectorstore-builder` | Récupère les événements et construit l'index FAISS |
| `api` | Lance l'API FastAPI |

L'API dépend du service `vectorstore-builder`, car elle a besoin de l'index FAISS pour répondre aux questions.

Les dossiers suivants sont montés comme volumes :

```text
data/
vectorstore/
```

Cela permet de conserver les données et l'index entre plusieurs lancements.

## Tests

Lancer tous les tests :

```bash
pytest
```

Lancer les tests avec couverture :

```bash
pytest --cov=src --cov=api
```

Les tests couvrent notamment :

- le preprocessing 
- le chunking 
- les embeddings 
- le vectorstore 
- l'API 
- l'évaluation

## Évaluation Du RAG

Le script `scripts/evaluate_rag.py` permet d'évaluer le système sur un jeu de questions.

Entrée :

```text
data/evaluation/test_questions.csv
```

Sortie :

```text
data/evaluation/evaluation_results.csv
```

Le script peut calculer :

- une similarité textuelle avec une réponse de référence 
- un exact match 
- des règles métier 
- des métriques DeepEval si elles sont disponibles

## Choix Techniques Importants

### Pourquoi utiliser un RAG ?

Les événements changent régulièrement. Un modèle de langage seul ne peut pas connaître l'état exact des événements disponibles dans OpenAgenda.

Le RAG permet de donner au modèle un contexte récent et contrôlé.

### Pourquoi utiliser FAISS ?

FAISS permet une recherche rapide par similarité vectorielle. C'est adapté à une recherche sémantique locale sur des textes d'événements.

### Pourquoi utiliser Mistral ?

Mistral est utilisé pour deux usages :

- transformer les textes en embeddings 
- générer les réponses finales

Cela garde une cohérence dans la chaîne IA et permet de travailler efficacement en français.

### Pourquoi utiliser FastAPI ?

FastAPI permet de créer une API claire, rapide et automatiquement documentée. Les modèles Pydantic valident les entrées et les sorties.

### Pourquoi utiliser Docker ?

Docker simplifie le lancement du projet sur une autre machine. Il limite les problèmes liés aux versions de Python ou aux dépendances installées localement.

## Limites Actuelles

Le projet est fonctionnel, mais certaines limites existent :

- la détection des questions futures repose sur des mots-clés 
- le filtrage par ville n'est pas encore strict dans la recherche principale 
- il n'y a pas encore de reranking avancé 
- l'index FAISS est local 
- la fraîcheur des données dépend de la dernière reconstruction de l'index 
- la qualité dépend fortement des données OpenAgenda 
- l'API ne propose pas encore d'authentification complète pour `/ask` 
- il n'y a pas encore d'interface utilisateur graphique

## Améliorations Possibles

Pistes d'amélioration :

- extraire automatiquement la ville, la date et le type d'événement depuis la question 
- ajouter un filtrage strict par métadonnées 
- ajouter un reranker après FAISS 
- mieux gérer les expressions temporelles comme `ce soir`, `samedi`, `la semaine prochaine` 
- automatiser la reconstruction régulière de l'index 
- ajouter une interface web 
- ajouter du monitoring et des logs structurés 
- ajouter un système de cache 
- améliorer les métriques d'évaluation 

## Commandes Utiles

```bash
# Vérifier l'environnement
python check_env.py

# Récupérer et nettoyer les événements
python -m scripts.fetch_openagenda

# Construire l'index FAISS
python -m scripts.build_vectorstore

# Tester le RAG directement
python -m scripts.ask_rag

# Lancer l'API
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Tester l'API manuellement
python -m scripts.api_test

# Lancer les tests
pytest

# Lancer avec Docker
docker compose up --build
```

## Résumé

Puls-Events RAG Chatbot est un proof of concept de chatbot intelligent spécialisé dans les événements culturels.

Le projet met en place une chaîne complète :

```text
collecte de données
→ nettoyage
→ indexation vectorielle
→ recherche sémantique
→ génération augmentée par contexte
→ API REST
→ évaluation
```

Le point central du projet est que le modèle ne répond pas seul : il s'appuie sur des événements retrouvés dans une base locale, ce qui rend les réponses plus fiables, plus vérifiables et plus adaptées au domaine métier.
```