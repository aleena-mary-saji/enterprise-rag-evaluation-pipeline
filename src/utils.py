import os
import yaml
import logging
from typing import List, Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads and returns yaml configuration."""
    if not os.path.exists(config_path):
        # Return sensible defaults if config is missing
        return {
            "data": {"docs_dir": "data/sample_docs", "db_dir": "data/vectordb", "collection_name": "rag_knowledge_base"},
            "chunking": {"method": "semantic", "min_chunk_size": 100, "max_chunk_size": 800, "similarity_threshold": 0.65},
            "embedding": {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"},
            "retrieval": {"dense_weight": 0.7, "sparse_weight": 0.3, "top_k_retrieved": 15, "top_k_reranked": 4, "reranker_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
            "llm": {"provider": "mock", "model_name": "llama3", "eval_model_name": "llama3", "api_base": "http://localhost:11434", "temperature": 0.0, "max_tokens": 800}
        }
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(level=logging.INFO):
    """Sets up unified logging formatting for stdout and file output."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("rag_pipeline.log", encoding="utf-8")
        ]
    )
    # Disable spammy logs from third party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# Let's import our classes locally to prevent circular imports if any
from src.ingestion import DocumentLoader, SemanticChunker, RecursiveCharacterChunker
from src.embeddings import EmbeddingEngine, VectorDBManager
from src.retrieval import HybridRetriever
from src.generation import LLMClient
from src.evaluation import RAGEvaluator

class RAGPipeline:
    """
    Facade class acting as the single point of orchestration for the RAG pipeline.
    Initializes and manages ingestion, vector indexing, hybrid search, 
    answer generation, and evaluation scoring.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        setup_logging()
        self.logger = logging.getLogger("RAGPipeline")
        self.logger.info("Initializing RAG Pipeline components...")
        
        # 1. Initialize Embeddings and Vector DB
        self.embedding_engine = EmbeddingEngine(
            model_name=self.config["embedding"]["model_name"],
            device=self.config["embedding"]["device"]
        )
        self.vector_db = VectorDBManager(
            db_dir=self.config["data"]["db_dir"],
            collection_name=self.config["data"]["collection_name"],
            embedding_engine=self.embedding_engine
        )
        
        # 2. Initialize Retriever
        self.retriever = HybridRetriever(
            vector_db=self.vector_db,
            reranker_model_name=self.config["retrieval"]["reranker_model_name"],
            dense_weight=self.config["retrieval"]["dense_weight"],
            sparse_weight=self.config["retrieval"]["sparse_weight"],
            device=self.config["embedding"]["device"]
        )
        
        # 3. Initialize LLM Client and Evaluator
        self.llm_client = LLMClient(
            provider=self.config["llm"]["provider"],
            model_name=self.config["llm"]["model_name"],
            api_base=self.config["llm"]["api_base"],
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"]
        )
        self.evaluator = RAGEvaluator(llm_client=self.llm_client)
        
        # 4. Initialize Chunkers
        # The SemanticChunker requires the embedding engine's document embedding method
        self.semantic_chunker = SemanticChunker(
            embed_fn=self.embedding_engine.embed_documents,
            min_chunk_size=self.config["chunking"]["min_chunk_size"],
            max_chunk_size=self.config["chunking"]["max_chunk_size"],
            similarity_threshold=self.config["chunking"]["similarity_threshold"]
        )
        self.recursive_chunker = RecursiveCharacterChunker(
            chunk_size=self.config["chunking"]["max_chunk_size"],
            chunk_overlap=int(self.config["chunking"]["max_chunk_size"] * 0.1)
        )
        
        self.logger.info("RAG Pipeline initialization completed successfully.")

    def ingest_file(self, file_path: str, chunk_method: str = "semantic") -> int:
        """
        Loads, chunks, and indexes a file into the vector store.
        
        Args:
            file_path: Path to txt, md, or pdf.
            chunk_method: "semantic" or "recursive".
            
        Returns:
            Number of chunks generated and indexed.
        """
        self.logger.info(f"Ingesting file: {file_path} using '{chunk_method}' chunking...")
        text = DocumentLoader.load(file_path)
        if not text.strip():
            self.logger.warning(f"File {file_path} is empty or failed to extract text.")
            return 0
            
        metadata = {
            "source_file": os.path.basename(file_path),
            "file_path": file_path,
        }
        
        # Select chunker
        if chunk_method == "semantic":
            chunks = self.semantic_chunker.chunk_text(text, metadata)
        else:
            chunks = self.recursive_chunker.chunk_text(text, metadata)
            
        if not chunks:
            self.logger.warning(f"No chunks generated for {file_path}.")
            return 0
            
        self.logger.info(f"Generated {len(chunks)} chunks. Indexing into vector store...")
        self.vector_db.add_documents(chunks)
        return len(chunks)

    def query(self, user_query: str, enable_rewrite: bool = False, run_eval: bool = True) -> Dict[str, Any]:
        """
        Processes a user query end-to-end:
        1. Optional query expansion/rewriting.
        2. Hybrid search & Cross-Encoder re-ranking.
        3. Contextual answer generation.
        4. Real-time evaluation of RAG Triad.
        
        Returns:
            Dictionary containing response, contexts, evaluation results, and trace logs.
        """
        self.logger.info(f"Processing query: '{user_query}'...")
        
        # 1. Query Expansion
        search_query = user_query
        if enable_rewrite:
            search_query = self.llm_client.rewrite_query(user_query)
            self.logger.info(f"Query expanded to: '{search_query}'")
            
        # 2. Hybrid Retrieval
        retrieval_output = self.retriever.retrieve(
            query_text=search_query,
            top_k_retrieved=self.config["retrieval"]["top_k_retrieved"],
            top_k_reranked=self.config["retrieval"]["top_k_reranked"]
        )
        
        retrieved_chunks = retrieval_output["results"]
        trace = retrieval_output["trace"]
        trace["expanded_query"] = search_query
        
        # 3. Answer Generation
        contexts = [c["text"] for c in retrieved_chunks]
        if not contexts:
            self.logger.warning("No relevant context found. Returning fallback answer.")
            answer = "I cannot find any relevant documents to answer your question."
            eval_output = {
                "faithfulness": {"score": 0.0, "reasoning": "No context available."},
                "answer_relevance": {"score": 0.0, "reasoning": "Answer is a fallback rejection."},
                "context_precision": {"score": 0.0, "reasoning": "Zero chunks retrieved."},
                "average_score": 0.0
            }
        else:
            answer = self.llm_client.generate_answer(user_query, contexts)
            
            # 4. Evaluation
            if run_eval:
                self.logger.info("Running LLM-as-a-Judge evaluations...")
                eval_output = self.evaluator.evaluate_all(
                    query=user_query,
                    answer=answer,
                    retrieved_chunks=retrieved_chunks
                )
            else:
                eval_output = {}
                
        return {
            "query": user_query,
            "search_query": search_query,
            "answer": answer,
            "retrieved_contexts": retrieved_chunks,
            "evaluation": eval_output,
            "trace": trace
        }
