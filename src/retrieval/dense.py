"""Dense retriever: embed the question with bge-m3, search Qdrant.

Assumes the ingestion step uploaded each chunk with a payload like
    {"text": "...", "source": "paper.pdf", "type": "pdf", "page": 7}
using the same embedding model, in a collection with one unnamed vector.
"""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        collection: str,
        url: str = "http://localhost:6333",
        embedding_model: str = "BAAI/bge-m3",
        device: str = "cpu",  # CPU keeps the 4 GB of VRAM free for Ollama
    ):
        self.collection = collection
        self.client = QdrantClient(url=url)
        self.model = SentenceTransformer(embedding_model, device=device)

    def search(self, question: str, k: int = 5) -> list[dict]:
        vector = self.model.encode(question, normalize_embeddings=True).tolist()
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            with_payload=True,
        ).points
        chunks = []
        for hit in hits:
            chunk = dict(hit.payload)
            chunk["score"] = hit.score
            chunk["chunk_id"] = hit.id
            chunks.append(chunk)
        return chunks