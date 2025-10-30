from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

class SearchEngine:
    def __init__(self, model_name: str, collection_name: str = "documents", distance: Distance = Distance.COSINE):
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.distance = distance
        self.vector_size = self.model.get_sentence_embedding_dimension()
        self._init_collection()
    
    def _init_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance)
        )
    
    def add_documents(self, documents: List[str], doc_ids: List[str] = None):
        embeddings = self.model.encode(documents, show_progress_bar=True)
        
        points = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = doc_ids[i] if doc_ids else str(i)
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={"text": doc, "doc_id": doc_id}
            ))
        
        self.client.upsert(collection_name=self.collection_name, points=points)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_vector = self.model.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        return [{"doc_id": r.payload["doc_id"], 
                 "score": r.score, 
                 "text": r.payload["text"]} for r in results]
