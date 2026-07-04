from unittest.mock import MagicMock, patch
from src.retrieval import HybridRetriever

@patch('src.retrieval.HybridRetriever._load_reranker')
def test_retrieval_tokenization(mock_load):
    """Verifies retrieval tokenizer parses text correctly."""
    mock_db = MagicMock()
    retriever = HybridRetriever(vector_db=mock_db, reranker_model_name="mock")
    
    tokens = retriever._tokenize("Advanced RAG, pipeline testing!")
    assert tokens == ["advanced", "rag,", "pipeline", "testing!"]

@patch('src.retrieval.HybridRetriever._load_reranker')
def test_bm25_scoring(mock_load):
    """Verifies BM25 sparse scoring identifies term matches and yields normalized outputs."""
    mock_db = MagicMock()
    retriever = HybridRetriever(vector_db=mock_db, reranker_model_name="mock")
    
    corpus = [
        "Quarterly subscription revenues increased by eight percent.",
        "The software architecture leverages a persistent client.",
        "Financial forecasting documents are stored in the database."
    ]
    
    # Query: "revenues"
    scores = retriever._get_bm25_scores("revenues", corpus)
    
    assert len(scores) == 3
    # Index 0 contains "revenues" -> score should be > 0.0 (and max score normalized to 1.0)
    assert scores[0] == 1.0
    # Index 1 and 2 do not contain the term -> score should be 0.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0

@patch('src.retrieval.HybridRetriever._load_reranker')
def test_hybrid_retrieval_fusion(mock_load):
    """Verifies hybrid search fuses results and cross-encoder re-ranks candidates."""
    mock_db = MagicMock()
    
    # Mock ChromaDB get() response representing index
    mock_db.collection.get.return_value = {
        "documents": [
            "Quarterly subscription revenues increased by eight percent.",
            "The software architecture leverages a persistent client.",
            "Some unrelated third text."
        ],
        "metadatas": [
            {"source_file": "financials.txt", "chunk_index": 0},
            {"source_file": "docs.txt", "chunk_index": 0},
            {"source_file": "dummy.txt", "chunk_index": 0}
        ],
        "ids": ["id1", "id2", "id3"]
    }
    
    # Mock vector database query response (dense)
    # Let's say id2 is slightly closer in dense vector space
    mock_db.query.return_value = [
        {"id": "id2", "text": "The software architecture leverages a persistent client.", "metadata": {"source_file": "docs.txt", "chunk_index": 0}, "score": 0.8},
        {"id": "id1", "text": "Quarterly subscription revenues increased by eight percent.", "metadata": {"source_file": "financials.txt", "chunk_index": 0}, "score": 0.5}
    ]
    
    retriever = HybridRetriever(vector_db=mock_db, reranker_model_name="mock")
    
    # Mock CrossEncoder predict: let's score id1 higher in re-ranking (e.g. 0.9 vs 0.1)
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.9, 0.1] # scores corresponding to RRF sorted candidates
    retriever.reranker = mock_reranker
    
    output = retriever.retrieve(query_text="revenues", top_k_retrieved=2, top_k_reranked=2)
    results = output["results"]
    
    assert len(results) == 2
    # Verify that id1 is re-ranked to the top spot (re-rank score 0.9) despite id2 being top in vector search
    assert results[0]["id"] == "id1"
    assert results[0]["rerank_score"] == 0.9
    assert results[1]["id"] == "id2"
    assert results[1]["rerank_score"] == 0.1
