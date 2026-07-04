# Enterprise RAG & LLM-as-a-Judge Evaluation Pipeline

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade **Retrieval-Augmented Generation (RAG)** pipeline integrated with an automated **LLM-as-a-Judge evaluation framework**. This system uses local open-source models to deliver domain-agnostic search answers while calculating real-time validation metrics to detect and prevent model hallucinations.

Designed to illustrate senior-level data science and machine learning engineering (LLMOps) principles, the repository showcases modular software design, advanced text chunking, hybrid retrieval architectures, and statistical evaluations.

---

## 🏗️ System Architecture

The pipeline is decoupled into discrete stages, wrapping all subsystems behind a clean **Facade Pattern** (`RAGPipeline`):

```
                                      [Ingestion Path]
                                             │
                                     [Document Loader]
                                             │
                                    [Semantic Chunker]
                                             │
                                   [Embedding Generator]
                                             │
                                    [(Vector DB: Chroma)]
                                             
                                       [Query Path]
                                             │
                                       [User Query]
                                             │
                                    [Query Expansion]
                                             │
                          ┌──────────────────┴──────────────────┐
                          ▼                                     ▼
                [Dense Vector Search]                 [Sparse BM25 Search]
                          │                                     │
                          └──────────────────┬──────────────────┘
                                             ▼
                               [Reciprocal Rank Fusion]
                                             │
                                 [Cross-Encoder Rerank]
                                             │
                                    [Answer Generator]
                                             │
                                   [LLM-as-a-Judge Audit]
                                  (Faithfulness / relevance)
```

### Key Engineering Pillars

1. **Semantic Chunking (`src/ingestion.py`)**: 
   Rather than using rigid character overlaps, text is parsed into sentences, embedded, and divided dynamically at points of semantic similarity drops. This preserves context boundaries and isolates thoughts cleanly, decreasing downstream model hallucinations.
2. **Hybrid Search & Fusion (`src/retrieval.py`)**:
   Combines semantic density (vector cosine similarity via ChromaDB) with lexical keyword matching (BM25 Okapi). The two candidate streams are merged using **Reciprocal Rank Fusion (RRF)**, mitigating bias from disparate score ranges.
3. **Cross-Encoder Re-Ranking (`src/retrieval.py`)**:
   The merged top-$K$ candidates are evaluated pairwise against the query using a HuggingFace Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`). This determines absolute question-context alignment, filtering irrelevant text before LLM context injection.
4. **LLM-as-a-Judge RAG Triad (`src/evaluation.py`)**:
   Evaluates performance in real-time without ground truth labels:
   * **Context Precision** (Search): Calculates the Mean Average Precision (MAP) of retrieved contexts using the LLM to judge individual chunk relevance.
   * **Faithfulness** (Hallucination audit): Extracts claims from the generated answer and verifies them against context facts.
   * **Answer Relevance** (Responsiveness): Rates how effectively the answer addresses the user's prompt on a normalized Likert rubric.

---

## 📁 Repository Structure

```
├── config/
│   └── config.yaml             # System & model hyperparameters
├── data/
│   ├── sample_docs/            # Directory to upload documents for search
│   └── vectordb/               # Persistence directory for ChromaDB
├── src/                        # Core Logic Package
│   ├── ingestion.py            # Document parsers and Semantic/Recursive splitters
│   ├── embeddings.py           # Embeddings Engine and Vector DB manager
│   ├── retrieval.py            # Hybrid search and Cross-Encoder re-ranker
│   ├── generation.py           # Ollama client and mock LLM wrapper
│   ├── evaluation.py           # LLM-as-a-Judge Triad auditing metrics
│   └── utils.py                # Logging configurations and Pipeline Orchestration
├── app/                        # Serving Layer
│   ├── api.py                  # FastAPI server hosting REST endpoints
│   └── main.py                 # Streamlit analytics dashboard and trace inspector
├── tests/                      # Testing Framework
│   ├── test_ingestion.py       # Unit tests for text chunkers
│   └── test_retrieval.py       # Unit tests for BM25 and RRF rank merging
├── Dockerfile                  # Production container definition
├── requirements.txt            # Application dependencies
└── .env.example                # Configuration blueprint
```

---

## ⚡ Quick Start

### Prerequisites
* Python 3.10 or higher
* (Optional) **Ollama** installed locally to run open-source models:
  * Download Ollama: https://ollama.com
  * Run the command: `ollama pull llama3` (or `phi3`, `mistral`)

---

### Setup Option A: Local Installation

1. **Clone the repository and navigate to the project root:**
   ```bash
   cd "Data Science Project"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation by running the test suite:**
   ```bash
   pytest
   ```

5. **Start the Streamlit Dashboard:**
   ```bash
   streamlit run app/main.py
   ```
   Open your browser to http://localhost:8501.

6. **(Optional) Run the FastAPI backend service:**
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
   ```
   Interactive Swagger documentation will be available at http://localhost:8000/docs.

---

### Setup Option B: Running with Docker

1. **Build the container:**
   ```bash
   docker build -t rag-evaluator .
   ```

2. **Run the container (exposing ports 8000 and 8501):**
   ```bash
   docker run -p 8501:8501 -p 8000:8000 rag-evaluator
   ```

## ⚙️ Configuration & Model Tweaking

All parameters are declared centrally in `config/config.yaml`.
* **Zero-Setup Dry Run**: By default, `llm.provider` is set to `"mock"`. This allows the interface to run instantly without downloading large models or starting Ollama, returning deterministic simulated outputs.
* **Running Local Models**: Set `llm.provider: "ollama"` and ensure Ollama is serving on port 11434. The system will automatically use the active local model (e.g. `llama3`) for generation and grading.
* **Paid Cloud LLMs**: Set `llm.provider` to `"openai"` or `"gemini"`. Supply your API credentials inside your `.env` file (`OPENAI_API_KEY` or `GEMINI_API_KEY`). The engine will invoke the corresponding endpoints (defaulting to `gpt-4o-mini` or `gemini-1.5-flash`) via REST.

---

## 🧪 Testing and Quality Control

Unit tests are written using `pytest`. Test cases mock out the machine learning models and database systems to ensure high coverage and fast local execution (sub-second):

```bash
# Run all tests
pytest -v
```

Tests evaluate:
* Cosine similarity calculations and boundary thresholds in `SemanticChunker`.
* Index character limits in `RecursiveCharacterChunker`.
* Score normalization and index matching in BM25 search.
* Candidate ranking overrides after Cross-Encoder re-evaluation.
