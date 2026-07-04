import os
import hashlib
import numpy as np
from typing import List, Dict, Any, Union

class EmbeddingEngine:
    """Wrapper around SentenceTransformers for local vector embeddings generation."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Lazy loads the HuggingFace sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name} on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Please install it using 'pip install sentence-transformers'."
            )
            
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text documents."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        # Convert numpy array to list of floats for ChromaDB compatibility
        return embeddings.tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """Generates embedding for a single search query."""
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()


class VectorDBManager:
    """Manages index creation, document storage, and vector retrieval using ChromaDB."""
    
    def __init__(self, db_dir: str, collection_name: str, embedding_engine: EmbeddingEngine):
        """
        Args:
            db_dir: Path to directory where vector database files should be persistent.
            collection_name: Name of the vector collection.
            embedding_engine: Instantiated EmbeddingEngine.
        """
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.embedding_engine = embedding_engine
        self.client = None
        self.collection = None
        self._init_db()
        
    def _init_db(self):
        """Initializes ChromaDB persistent client and collection."""
        try:
            import chromadb
            # Ensure folder exists
            os.makedirs(self.db_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.db_dir)
            
            # Use custom embedding function interface of chroma or handle embeddings ourselves.
            # We handle embedding generation ourselves to keep it clean and modular.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # Use cosine similarity distance
            )
        except ImportError:
            raise ImportError(
                "chromadb not installed. Please install it using 'pip install chromadb'."
            )
            
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Adds text chunks to the vector database collection.
        
        Args:
            chunks: List of dictionaries with 'text' and 'metadata' keys.
        """
        if not chunks:
            return
            
        texts = [c['text'] for c in chunks]
        metadatas = [self._flatten_metadata(c.get('metadata', {})) for c in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_engine.embed_documents(texts)
        
        # Generate unique deterministic IDs based on text hash
        ids = []
        for i, text in enumerate(texts):
            hasher = hashlib.md5()
            hasher.update(text.encode('utf-8'))
            source_file = metadatas[i].get('source_file', 'unknown')
            chunk_idx = metadatas[i].get('chunk_index', i)
            ids.append(f"{source_file}_{chunk_idx}_{hasher.hexdigest()[:8]}")
            
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"Successfully added {len(chunks)} chunks to vector database.")
        
    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        """ChromaDB metadatas must contain only simple types: str, int, float, or bool."""
        flat_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                flat_meta[k] = v
            elif isinstance(v, list):
                flat_meta[k] = ",".join(map(str, v))
            elif v is None:
                flat_meta[k] = "None"
            else:
                flat_meta[k] = str(v)
        return flat_meta

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB collection for top_k closest chunks.
        
        Returns:
            List of dictionary results containing 'text', 'metadata', and 'score'.
        """
        query_vector = self.embedding_engine.embed_query(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        formatted_results = []
        if results and results['documents'] and len(results['documents'][0]) > 0:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]
            
            for i in range(len(documents)):
                # Chroma distance for cosine is (1 - similarity).
                # Convert back to similarity score for intuitive presentation: similarity = 1 - distance
                similarity = 1.0 - distances[i]
                formatted_results.append({
                    "id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i],
                    "score": max(0.0, min(1.0, similarity)) # Clip to [0, 1]
                })
        return formatted_results
        
    def get_document_count(self) -> int:
        """Returns the number of documents/chunks indexed in the collection."""
        return self.collection.count()
        
    def reset(self):
        """Clears the current collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector database collection reset.")
