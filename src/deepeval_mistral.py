# Permet d'utiliser les annotations de type de facon plus souple.
from __future__ import annotations

# Importe os pour lire les variables d'environnement.
import os
# Importe Any pour typer une valeur dont le type exact peut varier.
from typing import Any

# Importe load_dotenv pour charger le fichier .env.
from dotenv import load_dotenv
# Importe la classe de base attendue par DeepEval pour definir un modele juge.
from deepeval.models import DeepEvalBaseLLM
# Importe le client officiel Mistral.
from mistralai.client import Mistral

# Charge les variables d'environnement depuis le fichier .env.
load_dotenv()


# Adapte Mistral au format attendu par DeepEval.
class DeepEvalMistralLLM(DeepEvalBaseLLM):
    # Initialise le modele juge utilise par DeepEval.
    def __init__(self, model_name: str = "mistral-small-latest") -> None:
        # Stocke le nom du modele Mistral a utiliser.
        self.model_name = model_name

        # Recupere la cle API Mistral depuis l'environnement.
        api_key = os.getenv("MISTRAL_API_KEY")
        # Si la cle est absente, DeepEval ne pourra pas appeler Mistral.
        if not api_key:
            # Signale clairement la variable manquante.
            raise RuntimeError("MISTRAL_API_KEY est absente du fichier .env")

        # Cree le client Mistral avec la cle API.
        self.client = Mistral(api_key=api_key)

    # Methode demandee par DeepEval pour charger le modele.
    def load_model(self) -> Any:
        # Retourne le client Mistral deja initialise.
        return self.client

    # Genere une reponse du modele juge a partir d'un prompt.
    def generate(self, prompt: str) -> str:
        # Appelle l'API chat de Mistral avec le prompt fourni par DeepEval.
        response = self.client.chat.complete(
            # Nom du modele Mistral utilise comme juge.
            model=self.model_name,
            # Liste des messages envoyes au modele.
            messages=[
                {
                    # Le message vient de l'utilisateur.
                    "role": "user",
                    # Contenu du prompt a evaluer.
                    "content": prompt,
                }
            ],
            # Temperature a 0 pour rendre le jugement le plus stable possible.
            temperature=0,
        )

        # Retourne le texte genere par le modele.
        return response.choices[0].message.content

    # Version asynchrone demandee par DeepEval.
    async def a_generate(self, prompt: str) -> str:
        # Reutilise la version synchrone pour eviter de dupliquer la logique.
        return self.generate(prompt)

    # Retourne le nom du modele utilise.
    def get_model_name(self) -> str:
        # Renvoie le nom stocke a l'initialisation.
        return self.model_name