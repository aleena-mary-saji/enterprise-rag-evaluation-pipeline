import os
import time
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# Add parent directory to path so we can import src modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import RAGPipeline

# Set Page Config
st.set_page_config(
    page_title="Enterprise RAG Evaluator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek CSS for Aesthetics
st.markdown("""
<style>
    /* Sleek Title and Headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #7E22CE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #71717A;
        margin-bottom: 2rem;
    }
    
    /* Metrics Styling */
    .metric-card {
        background-color: #1E1B4B;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #4338CA;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #818CF8;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #A5F3FC;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Highlight containers */
    .source-box {
        border-left: 4px solid #818CF8;
        padding: 10px 15px;
        background-color: #1F2937;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    
    .score-badge {
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State for query history
if "history" not in st.session_state:
    st.session_state.history = []

# Cache pipeline loader so it only runs once
@st.cache_resource
def load_pipeline():
    return RAGPipeline(config_path="config/config.yaml")

try:
    pipeline = load_pipeline()
    pipeline_loaded = True
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    pipeline_loaded = False

# Sidebar Panels
st.sidebar.markdown("<h2 style='text-align: center; color: #818CF8;'>System Control Center</h2>", unsafe_allow_html=True)

if pipeline_loaded:
    # 1. Document Index Status
    chunk_count = pipeline.vector_db.get_document_count()
    st.sidebar.info(f"📁 **Vector Index Status**: {chunk_count} Chunks Indexed")
    
    # 2. Ingestion Panel
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Ingest Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload reference documents (TXT, MD, PDF)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )
    
    chunk_method = st.sidebar.selectbox("Chunking Strategy", ["semantic", "recursive"], index=0)
    
    if st.sidebar.button("Ingest Files", use_container_width=True) and uploaded_files:
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        progress_bar = st.sidebar.progress(0.0)
        status_text = st.sidebar.empty()
        
        total_indexed_chunks = 0
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            try:
                indexed = pipeline.ingest_file(temp_path, chunk_method=chunk_method)
                total_indexed_chunks += indexed
            except Exception as ex:
                st.sidebar.error(f"Error indexing {uploaded_file.name}: {ex}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            progress_bar.progress(float((idx + 1) / len(uploaded_files)))
            
        status_text.text("Ingestion completed!")
        st.sidebar.success(f"Indexed {total_indexed_chunks} chunks using '{chunk_method}' chunking.")
        time.sleep(2)
        st.rerun()

    # 3. Settings Panel
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pipeline Configs")
    st.sidebar.text(f"Embedder: {pipeline.embedding_engine.model_name.split('/')[-1]}")
    st.sidebar.text(f"Reranker: {pipeline.retriever.reranker_model_name.split('/')[-1]}")
    st.sidebar.text(f"LLM Engine: {pipeline.llm_client.provider.upper()} ({pipeline.llm_client.model_name})")
    
    # 4. System Actions
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Knowledge Base", type="primary", use_container_width=True):
        pipeline.vector_db.reset()
        st.sidebar.success("Database cleared successfully.")
        st.session_state.history = []
        time.sleep(1)
        st.rerun()

# Main Application Board
st.markdown("<div class='main-title'>Enterprise RAG & LLM-as-a-Judge Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Production-grade Retrieval-Augmented Generation with automatic real-time evaluation using the RAG Triad.</div>", unsafe_allow_html=True)

if not pipeline_loaded:
    st.warning("Please check system configuration and ensure prerequisites are met.")
else:
    # Set up layout tabs
    tab_chat, tab_retrieval, tab_judge, tab_analytics = st.tabs([
        "💬 Chat Sandbox", 
        "🔍 Retrieval Inspector", 
        "⚖️ LLM Judge Audit", 
        "📊 System Analytics"
    ])
    
    # Session current query store
    current_result = None
    
    with tab_chat:
        st.markdown("### Chat with your Knowledge Base")
        
        # User input query
        query_input = st.text_input(
            "Ask a question based on your uploaded documentation:",
            placeholder="e.g., What is the company's Q4 subscription revenue?",
            key="user_query_field"
        )
        
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            enable_rewrite = st.checkbox("Enable Query Expansion (LLM rewriting)", value=False)
        with col_opt2:
            run_eval = st.checkbox("Run LLM-as-a-Judge Evaluations", value=True)
            
        if st.button("Submit Query", type="primary") and query_input:
            if chunk_count == 0:
                st.warning("The vector database is currently empty. Please upload and ingest documents in the sidebar first!")
            else:
                with st.spinner("Processing hybrid search and generating answer..."):
                    # Execute RAG query
                    res = pipeline.query(query_input, enable_rewrite=enable_rewrite, run_eval=run_eval)
                    current_result = res
                    
                    # Log to history
                    log_entry = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "query": query_input,
                        "expanded_query": res["search_query"],
                        "answer": res["answer"],
                        "contexts": res["retrieved_contexts"],
                        "evaluation": res["evaluation"],
                        "trace": res["trace"]
                    }
                    st.session_state.history.append(log_entry)
        
        # Show current answer if available
        if current_result:
            st.markdown("#### Response")
            st.markdown(current_result["answer"])
            
            # Show sources
            st.markdown("#### Retrieved Sources")
            for idx, doc in enumerate(current_result["retrieved_contexts"]):
                with st.expander(f"Context {idx+1}: {doc['metadata'].get('source_file', 'Unknown')} (Score: {doc['rerank_score']:.4f})"):
                    st.markdown(f"**Snippet:**\n{doc['text']}")
                    st.json(doc['metadata'])
        elif st.session_state.history:
            # Show last history item
            last = st.session_state.history[-1]
            st.markdown("#### Response")
            st.markdown(last["answer"])
            
            st.markdown("#### Retrieved Sources")
            for idx, doc in enumerate(last["contexts"]):
                score_val = doc.get('rerank_score', doc.get('score', 0.0))
                with st.expander(f"Context {idx+1}: {doc['metadata'].get('source_file', 'Unknown')} (Score: {score_val:.4f})"):
                    st.markdown(f"**Snippet:**\n{doc['text']}")
                    st.json(doc['metadata'])
                    
    with tab_retrieval:
        st.markdown("### Retrieval Trace Inspector")
        st.markdown("Understand how the hybrid search and cross-encoder re-ranking optimize candidate chunks.")
        
        target_log = current_result if current_result else (st.session_state.history[-1] if st.session_state.history else None)
        
        if not target_log:
            st.info("Submit a query to inspect the retrieval path.")
        else:
            trace = target_log["trace"]
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("Total Indexed Chunks", trace.get("total_indexed_chunks", 0))
            with col_t2:
                st.metric("Dense Vector Candidates", trace.get("dense_candidates_count", 0))
            with col_t3:
                st.metric("Sparse BM25 Candidates", trace.get("sparse_candidates_count", 0))
                
            st.markdown("#### Expanded Search Terms")
            st.code(target_log["search_query"])
            
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.markdown("##### Top Hits: Dense Vector Search (Semantic)")
                for hit in trace.get("dense_top_hits", []):
                    st.markdown(f"- **ID**: `{hit['id']}`  \n  *Score*: `{hit['score']:.4f}`  \n  *Text*: {hit['text']}")
            with col_h2:
                st.markdown("##### Top Hits: Sparse BM25 Search (Keyword)")
                for hit in trace.get("sparse_top_hits", []):
                    st.markdown(f"- **ID**: `{hit['id']}`  \n  *Score*: `{hit['score']:.4f}`  \n  *Text*: {hit['text']}")
                    
            st.markdown("---")
            st.markdown("#### Final Re-Ranked Context (Cross-Encoder Selection)")
            st.markdown("The top fused candidates are re-scored by the Cross-Encoder model to determine absolute query-chunk semantic alignment:")
            
            ranks_df = []
            for idx, doc in enumerate(target_log.get("contexts", target_log.get("retrieved_contexts", []))):
                ranks_df.append({
                    "Rank": idx + 1,
                    "Source": doc['metadata'].get('source_file', 'Unknown'),
                    "Index": doc['metadata'].get('chunk_index', 0),
                    "Vector Score": f"{doc.get('dense_score', 0.0):.4f}",
                    "BM25 Score": f"{doc.get('sparse_score', 0.0):.4f}",
                    "RRF Fusion Score": f"{doc.get('rrf_score', 0.0):.6f}",
                    "Cross-Encoder Score": f"{doc.get('rerank_score', 0.0):.4f}"
                })
            
            if ranks_df:
                st.table(pd.DataFrame(ranks_df))
                
    with tab_judge:
        st.markdown("### LLM-as-a-Judge Audit")
        st.markdown("Real-time automated evaluation metrics measuring LLM grounding and search retrieval precision.")
        
        target_log = current_result if current_result else (st.session_state.history[-1] if st.session_state.history else None)
        
        if not target_log or not target_log.get("evaluation"):
            st.info("Submit a query with 'Run LLM-as-a-Judge Evaluations' enabled to view scores.")
        else:
            eval_data = target_log["evaluation"]
            
            # Displays the RAG Triad
            col_e1, col_e2, col_e3 = st.columns(3)
            
            with col_e1:
                f_score = eval_data.get("faithfulness", {}).get("score", 0.0)
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-lbl'>😇 Faithfulness</div>"
                    f"<div class='metric-val'>{f_score:.2f}</div>"
                    f"<div style='color: #71717A; font-size: 0.85rem; margin-top: 0.5rem;'>Is the answer grounded in the context?</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
            with col_e2:
                r_score = eval_data.get("answer_relevance", {}).get("score", 0.0)
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-lbl'>🎯 Answer Relevance</div>"
                    f"<div class='metric-val'>{r_score:.2f}</div>"
                    f"<div style='color: #71717A; font-size: 0.85rem; margin-top: 0.5rem;'>Does the answer address the question?</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
            with col_e3:
                p_score = eval_data.get("context_precision", {}).get("score", 0.0)
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-lbl'>⚖️ Context Precision</div>"
                    f"<div class='metric-val'>{p_score:.2f}</div>"
                    f"<div style='color: #71717A; font-size: 0.85rem; margin-top: 0.5rem;'>Were relevant contexts ranked at top?</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
            st.markdown("---")
            st.markdown("#### Judge Reasoning Logs")
            
            with st.expander("Faithfulness Auditor Claims Breakdown"):
                st.write(f"**Summary Auditor Rationale:** {eval_data.get('faithfulness', {}).get('reasoning', '')}")
                claims = eval_data.get("faithfulness", {}).get("claims", [])
                if claims:
                    st.table(pd.DataFrame(claims))
                    
            with st.expander("Answer Relevance Audit Detail"):
                st.write(f"**Auditor Rationale:** {eval_data.get('answer_relevance', {}).get('reasoning', '')}")
                
            with st.expander("Context Precision (MAP calculation) Audit Detail"):
                st.write(f"**Auditor Rationale:** {eval_data.get('context_precision', {}).get('reasoning', '')}")
                rel_vector = eval_data.get('context_precision', {}).get('relevance_vector', [])
                st.write(f"**Relevance Vector (Chunk order: 1 to K)**: `{rel_vector}`")
                
    with tab_analytics:
        st.markdown("### System Performance Analytics")
        
        if not st.session_state.history:
            st.info("Analytics will display once you perform multiple queries in the chat sandbox.")
        else:
            # Build data frame of session scores
            rows = []
            for idx, item in enumerate(st.session_state.history):
                eval_data = item.get("evaluation", {})
                if eval_data:
                    rows.append({
                        "Query Index": idx + 1,
                        "Query": item["query"][:20] + "...",
                        "Faithfulness": eval_data.get("faithfulness", {}).get("score", 0.0),
                        "Answer Relevance": eval_data.get("answer_relevance", {}).get("score", 0.0),
                        "Context Precision": eval_data.get("context_precision", {}).get("score", 0.0),
                        "Average RAG Triad": eval_data.get("average_score", 0.0)
                    })
                    
            if not rows:
                st.info("No queries found with evaluation data enabled.")
            else:
                df = pd.DataFrame(rows)
                st.markdown("#### Session Score Progression")
                
                # Chart
                df_melted = df.melt(id_vars=["Query Index", "Query"], value_vars=["Faithfulness", "Answer Relevance", "Context Precision", "Average RAG Triad"], var_name="Metric", value_name="Score")
                fig = px.line(
                    df_melted, 
                    x="Query Index", 
                    y="Score", 
                    color="Metric", 
                    markers=True,
                    title="Evaluation Score Tracking over Session Queries",
                    color_discrete_sequence=["#F87171", "#34D399", "#60A5FA", "#C084FC"]
                )
                fig.update_layout(yaxis_range=[0, 1.05])
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Session Queries Summary")
                st.table(df)
