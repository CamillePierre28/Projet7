# Importe tempfile pour creer un dossier temporaire pendant le test.
import tempfile

# Importe FAISS, le vectorstore utilise pour indexer et rechercher les documents.
from langchain_community.vectorstores import FAISS
# Importe la classe de base Embeddings pour creer un faux modele d'embeddings.
from langchain_core.embeddings import Embeddings
# Importe Document, le format de document utilise par LangChain.
from langchain_core.documents import Document


# Cree une fausse classe d'embeddings pour tester FAISS sans appeler une vraie API.
class FakeEmbeddings(Embeddings):
    # Transforme une liste de textes en vecteurs numeriques simples.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Retourne un vecteur base sur la longueur de chaque texte.
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    # Transforme une seule question en vecteur numerique simple.
    def embed_query(self, text: str) -> list[float]:
        # Retourne un vecteur base sur la longueur de la question.
        return [float(len(text)), 1.0, 0.0]


# Teste que FAISS peut creer un index et faire une recherche.
def test_faiss_index_creation_and_search():
    # Cree deux documents de test avec un texte et des metadonnees.
    docs = [
        # Premier document, lie a Toulouse.
        Document(
            # Texte principal du document.
            page_content="Concert de jazz a Toulouse",
            # Informations associees au document.
            metadata={"uid": "1", "city": "Toulouse"},
        ),
        # Deuxieme document, lie a Montpellier.
        Document(
            # Texte principal du document.
            page_content="Exposition de peinture a Montpellier",
            # Informations associees au document.
            metadata={"uid": "2", "city": "Montpellier"},
        ),
    ]

    # Instancie le faux modele d'embeddings.
    embeddings = FakeEmbeddings()

    # Cree un index FAISS a partir des documents et des faux embeddings.
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Cherche le document le plus proche de la requete.
    results = vectorstore.similarity_search("jazz Toulouse", k=1)

    # Verifie qu'un seul resultat est retourne.
    assert len(results) == 1
    # Verifie que le resultat contient bien un identifiant dans ses metadonnees.
    assert "uid" in results[0].metadata


# Teste que l'index FAISS peut etre sauvegarde puis recharge.
def test_faiss_save_and_load():
    # Cree une liste contenant un document de test.
    docs = [
        # Document simple pour tester la sauvegarde.
        Document(
            # Texte principal du document.
            page_content="Festival culturel en Occitanie",
            # Metadonnee minimale du document.
            metadata={"uid": "1"},
        )
    ]

    # Instancie le faux modele d'embeddings.
    embeddings = FakeEmbeddings()

    # Cree un index FAISS a partir du document.
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Cree un dossier temporaire supprime automatiquement apres le test.
    with tempfile.TemporaryDirectory() as tmpdir:
        # Sauvegarde l'index FAISS dans le dossier temporaire.
        vectorstore.save_local(tmpdir)

        # Recharge l'index FAISS depuis le dossier temporaire.
        loaded = FAISS.load_local(
            # Chemin du dossier ou l'index a ete sauvegarde.
            tmpdir,
            # Meme modele d'embeddings que celui utilise a la creation.
            embeddings,
            # Autorise LangChain a recharger les fichiers locaux de l'index.
            allow_dangerous_deserialization=True,
        )

        # Lance une recherche dans l'index recharge.
        results = loaded.similarity_search("culture Occitanie", k=1)

        # Verifie qu'un resultat est retourne.
        assert len(results) == 1
        # Verifie que le document retrouve est bien celui attendu.
        assert results[0].metadata["uid"] == "1"