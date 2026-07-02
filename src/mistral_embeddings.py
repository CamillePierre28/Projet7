# Embeddings Mistral compatibles LangChain

# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe os pour lire les variables d'environnement.
import os
# Importe sleep pour attendre entre deux appels API.
from time import sleep

# Importe load_dotenv pour charger le fichier .env.
from dotenv import load_dotenv
# Importe la classe de base Embeddings attendue par LangChain.
from langchain_core.embeddings import Embeddings
# Importe le client officiel Mistral.
from mistralai.client import Mistral

# Charge les variables d'environnement depuis le fichier .env.
load_dotenv()


# Cree une classe d'embeddings Mistral compatible avec LangChain.
class MistralEmbeddings(Embeddings):
    # Initialise la classe avec le modele, la taille des lots et les retries.
    def __init__(
        self,
        model: str = "mistral-embed",
        batch_size: int | None = None,
        sleep_seconds: float | None = None,
        max_retries: int = 6,
    ) -> None:
        # Stocke le nom du modele d'embedding a utiliser.
        self.model = model
        # Utilise la taille de batch fournie ou celle definie dans le .env.
        self.batch_size = batch_size or int(
            os.getenv("MISTRAL_EMBED_BATCH_SIZE", "8")
        )
        # Utilise le temps d'attente fourni ou celui defini dans le .env.
        self.sleep_seconds = sleep_seconds or float(
            os.getenv("MISTRAL_EMBED_SLEEP_SECONDS", "2")
        )
        # Stocke le nombre maximal de tentatives en cas de rate limit.
        self.max_retries = max_retries

        # Recupere la cle API Mistral depuis l'environnement.
        api_key = os.getenv("MISTRAL_API_KEY")

        # Si la cle API est absente, le programme ne peut pas appeler Mistral.
        if not api_key:
            # Signale clairement la variable manquante.
            raise RuntimeError("MISTRAL_API_KEY est absente du fichier .env")

        # Cree le client Mistral avec la cle API.
        self.client = Mistral(api_key=api_key)

    # Envoie un lot de textes a Mistral pour obtenir leurs embeddings.
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Essaie plusieurs fois en cas de rate limit.
        for attempt in range(self.max_retries):
            # Tente l'appel API.
            try:
                # Demande a Mistral de creer les embeddings du lot de textes.
                response = self.client.embeddings.create(
                    # Nom du modele d'embedding.
                    model=self.model,
                    # Textes a vectoriser.
                    inputs=texts,
                )

                # Extrait les vecteurs numeriques depuis la reponse Mistral.
                return [item.embedding for item in response.data]

            # Capture les erreurs pour gerer les limites d'API.
            except Exception as exc:
                # Convertit l'erreur en texte pour l'analyser simplement.
                message = str(exc)

                # Si l'erreur n'est pas un rate limit, on la remonte directement.
                if "429" not in message and "rate" not in message.lower():
                    # Relance l'erreur originale.
                    raise

                # Calcule un temps d'attente progressif avant de reessayer.
                wait_time = self.sleep_seconds * (2 ** attempt)

                # Affiche un message pour expliquer pourquoi le script attend.
                print(
                    f"Rate limit Mistral détecté. "
                    f"Nouvelle tentative dans {wait_time:.1f}s..."
                )

                # Attend avant la prochaine tentative.
                sleep(wait_time)

        # Si toutes les tentatives echouent, on arrete avec une erreur claire.
        raise RuntimeError("Trop d'échecs après rate limit Mistral")

    # Methode appelee par LangChain pour vectoriser plusieurs documents.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Cree une liste vide pour stocker tous les embeddings.
        embeddings = []

        # Calcule le nombre total de textes a vectoriser.
        total = len(texts)

        # Parcourt les textes par lots pour eviter d'envoyer trop de donnees d'un coup.
        for start in range(0, total, self.batch_size):
            # Recupere un batch de textes.
            batch = texts[start : start + self.batch_size]

            # Vectorise le batch avec Mistral.
            vectors = self._embed_batch(batch)
            # Ajoute les vecteurs obtenus a la liste globale.
            embeddings.extend(vectors)

            # Calcule combien de textes ont deja ete traites.
            done = min(start + self.batch_size, total)
            # Affiche l'avancement de la vectorisation.
            print(f"{done}/{total} chunks vectorisés")

            # Attend un peu entre les appels pour limiter les risques de rate limit.
            sleep(self.sleep_seconds)

        # Retourne tous les embeddings calcules.
        return embeddings

    # Methode appelee par LangChain pour vectoriser une seule question utilisateur.
    def embed_query(self, text: str) -> list[float]:
        # Vectorise la question comme un batch contenant un seul texte.
        return self._embed_batch([text])[0]