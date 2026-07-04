import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from src.utils import RAGPipeline

app = FastAPI(
    title="Enterprise RAG Evaluation API",
    description="Backend API serving hybrid retrieval, document ingestion, and real-time LLM-as-a-Judge evaluations.",
    version="1.0.0"
)

# Initialize the pipeline singleton
# We specify default config path
pipeline = RAGPipeline(config_path="config/config.yaml")

class QueryRequest(BaseModel):
    query: str
    enable_rewrite: Optional[bool] = False
    run_eval: Optional[bool] = True

class QueryResponse(BaseModel):
    query: str
    search_query: str
    answer: str
    retrieved_contexts: List[Dict[str, Any]]
    evaluation: Dict[str, Any]
    trace: Dict[str, Any]

@app.get("/")
def read_root():
    return {
        "status": "online",
        "vector_store_collection": pipeline.vector_db.collection_name,
        "indexed_chunks": pipeline.vector_db.get_document_count(),
        "llm_provider": pipeline.llm_client.provider,
        "embedding_model": pipeline.embedding_engine.model_name
    }

@app.post("/ingest", response_model=Dict[str, Any])
def ingest_file(file: UploadFile = File(...), chunk_method: str = "semantic"):
    if chunk_method not in ["semantic", "recursive"]:
        raise HTTPException(status_code=400, detail="chunk_method must be 'semantic' or 'recursive'")
        
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save file to temp path
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run ingestion
        chunk_count = pipeline.ingest_file(temp_path, chunk_method=chunk_method)
        
        return {
            "filename": file.filename,
            "chunk_method": chunk_method,
            "status": "success",
            "chunks_indexed": chunk_count,
            "total_indexed_chunks": pipeline.vector_db.get_document_count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    try:
        result = pipeline.query(
            user_query=request.query,
            enable_rewrite=request.enable_rewrite,
            run_eval=request.run_eval
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/reset", response_model=Dict[str, Any])
def reset_database():
    try:
        pipeline.vector_db.reset()
        return {
            "status": "success",
            "message": "Vector database reset completed.",
            "total_indexed_chunks": pipeline.vector_db.get_document_count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Serve API locally on port 8000
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
