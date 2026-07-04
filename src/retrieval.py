import numpy as np
from typing import List, Dict, Any
from src.embeddings import VectorDBManager

class HybridRetriever:
    """
    Retrieves document chunks using a hybrid approach combining Dense Vector Search 
    and Sparse BM25 keyword search, fused via Reciprocal Rank Fusion (RRF) or 
    Weighted Score Fusion, and finally re-ranked using a Cross-Encoder model.
    """
    
    def __init__(
        self,
        vector_db: VectorDBManager,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        device: str = "cpu"
    ):
        """
        Args:
            vector_db: Instantiated VectorDBManager.
            reranker_model_name: HuggingFace model name for Cross-Encoder re-ranking.
            dense_weight: Weight for dense scores if using weighted fusion.
            sparse_weight: Weight for sparse scores if using weighted fusion.
            device: Computing device ('cpu' or 'cuda').
        """
        self.vector_db = vector_db
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.device = device
        self.reranker_model_name = reranker_model_name
        self.reranker = None
        self._load_reranker()
        
    def _load_reranker(self):
        """Lazy loads the Cross-Encoder model for re-ranking."""
        try:
            from sentence_transformers import CrossEncoder
            print(f"Loading Cross-Encoder re-ranking model: {self.reranker_model_name}...")
            self.reranker = CrossEncoder(self.reranker_model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Please install it using 'pip install sentence-transformers'."
            )
            
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25 (lowercases and splits words)."""
        return text.lower().replace(r'[^\w\s]', '').split()

    def _get_bm25_scores(self, query: str, documents: List[str]) -> List[float]:
        """Calculates BM25 scores for a query across all indexed documents."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[Warning] rank_bm25 not installed. Sparse BM25 search will return 0.0 scores.")
            return [0.0] * len(documents)
            
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        tokenized_query = self._tokenize(query)
        
        # If corpus is empty or query is empty, return zeros
        if not tokenized_corpus or not tokenized_query:
            return [0.0] * len(documents)
            
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
        
        # Normalize scores to [0, 1] range to allow fusion with dense scores
        max_score = max(scores) if len(scores) > 0 else 0
        if max_score > 0:
            scores = [float(s / max_score) for s in scores]
        else:
            scores = [0.0] * len(documents)
            
        return scores

    def retrieve(self, query_text: str, top_k_retrieved: int = 15, top_k_reranked: int = 4) -> Dict[str, Any]:
        """
        Runs the full hybrid search and re-ranking pipeline:
        1. Dense Vector Search
        2. Sparse BM25 Search
        3. Reciprocal Rank Fusion (RRF) to merge candidate lists
        4. Cross-Encoder Re-ranking on the top candidates
        
        Returns:
            Dictionary with keys 'results' (final re-ranked list) and 'metadata' (trace details).
        """
        # Fetch all indexed items from ChromaDB to populate BM25 corpus
        all_items = self.vector_db.collection.get()
        all_docs = all_items.get('documents', [])
        all_metadatas = all_items.get('metadatas', [])
        all_ids = all_items.get('ids', [])
        
        total_indexed = len(all_docs)
        if total_indexed == 0:
            return {"results": [], "metadata": {"trace": "No documents indexed in database."}}
            
        # 1. Dense Retrieval (Cosine Similarity)
        # Pull slightly more than top_k_retrieved to have enough overlap for hybrid fusion
        dense_results = self.vector_db.query(query_text, top_k=min(total_indexed, top_k_retrieved * 2))
        dense_ranks = {item['id']: i for i, item in enumerate(dense_results)}
        dense_scores = {item['id']: item['score'] for item in dense_results}
        
        # 2. Sparse Retrieval (BM25)
        bm25_scores_list = self._get_bm25_scores(query_text, all_docs)
        
        # Build candidate maps
        sparse_scores = {}
        for idx, doc_id in enumerate(all_ids):
            if bm25_scores_list[idx] > 0.0:
                sparse_scores[doc_id] = bm25_scores_list[idx]
                
        # Sort sparse results to get ranks
        sorted_sparse_ids = sorted(sparse_scores.keys(), key=lambda x: sparse_scores[x], reverse=True)
        sparse_ranks = {doc_id: i for i, doc_id in enumerate(sorted_sparse_ids[:top_k_retrieved * 2])}

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF score = sum(1 / (k + rank))
        # We use standard constant k = 60
        k_rrf = 60
        rrf_scores = {}
        all_candidates = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        
        for cand_id in all_candidates:
            dense_rank = dense_ranks.get(cand_id, None)
            sparse_rank = sparse_ranks.get(cand_id, None)
            
            score = 0.0
            if dense_rank is not None:
                score += 1.0 / (k_rrf + dense_rank)
            if sparse_rank is not None:
                score += 1.0 / (k_rrf + sparse_rank)
                
            rrf_scores[cand_id] = score
            
        # Sort candidates by RRF score
        sorted_candidates = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_candidates = sorted_candidates[:top_k_retrieved]
        
        # Resolve document texts and metadata for the top candidates
        candidate_docs = []
        for cand_id in top_candidates:
            db_idx = all_ids.index(cand_id)
            candidate_docs.append({
                "id": cand_id,
                "text": all_docs[db_idx],
                "metadata": all_metadatas[db_idx],
                "dense_score": dense_scores.get(cand_id, 0.0),
                "sparse_score": sparse_scores.get(cand_id, 0.0),
                "rrf_score": rrf_scores.get(cand_id, 0.0)
            })

        # 4. Cross-Encoder Re-ranking
        final_results = []
        if candidate_docs and self.reranker:
            pairs = [[query_text, doc['text']] for doc in candidate_docs]
            # Predict scores (higher = more relevant)
            rerank_scores = self.reranker.predict(pairs)
            
            # Add rerank scores
            for i, doc in enumerate(candidate_docs):
                # Sigmoid normalization if output is raw logits, or keep as float
                doc['rerank_score'] = float(rerank_scores[i])
                
            # Sort by rerank score descending
            final_results = sorted(candidate_docs, key=lambda x: x['rerank_score'], reverse=True)
        else:
            final_results = candidate_docs
            
        # Slice to final top_k_reranked
        top_final_results = final_results[:top_k_reranked]
        
        # Construct retrieval trace for dashboard inspection
        trace = {
            "total_indexed_chunks": total_indexed,
            "dense_candidates_count": len(dense_results),
            "sparse_candidates_count": len(sparse_scores),
            "merged_candidates_count": len(all_candidates),
            "dense_top_hits": [{"id": d['id'], "text": d['text'][:60] + "...", "score": d['score']} for d in dense_results[:3]],
            "sparse_top_hits": [{"id": doc_id, "text": all_docs[all_ids.index(doc_id)][:60] + "...", "score": sparse_scores[doc_id]} for doc_id in sorted_sparse_ids[:3]]
        }
        
        return {
            "results": top_final_results,
            "trace": trace
        }
