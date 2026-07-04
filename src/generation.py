import json
import requests
from typing import List, Dict, Any

class LLMClient:
    """
    Orchestrates query translation and generation using local open-source LLMs 
    (via Ollama's HTTP API) with a robust offline mock fallback.
    """
    
    def __init__(
        self,
        provider: str = "mock",
        model_name: str = "llama3",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 800
    ):
        """
        Args:
            provider: LLM engine choice ("ollama" or "mock").
            model_name: Local model tag in Ollama (e.g., 'llama3', 'mistral', 'phi3').
            api_base: Endpoint for Ollama service.
            temperature: Randomness control (0.0 for factual/evaluation consistency).
            max_tokens: Limit on generated tokens.
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Helper to invoke Ollama via REST API."""
        url = f"{self.api_base}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(url, json=payload, timeout=25)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f"[Ollama Error] Failed to connect: {e}. Falling back to mock generator.")
            # Gracefully degrade to mock
            return self._call_mock(prompt, system_prompt)

    def _call_mock(self, prompt: str, system_prompt: str = None) -> str:
        """
        Offline deterministic generator. Dynamically answers using context if provided in prompt.
        """
        # Attempt to extract context and query from the prompt
        prompt_lower = prompt.lower()
        query = "What is this document about?"
        
        # Simple extraction logic for query/context structure
        if "query:" in prompt_lower:
            parts = prompt.split("Query:")
            if len(parts) > 1:
                query = parts[1].split("\n")[0].strip()
        elif "question:" in prompt_lower:
            parts = prompt.split("Question:")
            if len(parts) > 1:
                query = parts[1].split("\n")[0].strip()
                
        # Look for Context block
        context_str = ""
        if "context:" in prompt_lower:
            parts = prompt.split("Context:")
            if len(parts) > 1:
                context_str = parts[1].split("---")[0].strip()
                
        # Parse query keyword to make answer mock-dynamic
        q_clean = query.lower()
        
        # Build answer based on context if available
        if context_str:
            sentences = [s.strip() for s in context_str.split("\n") if len(s.strip()) > 30]
            if sentences:
                summary_points = sentences[:2]
                res = f"Based on the provided documentation, here are the key findings:\n"
                for i, s in enumerate(summary_points):
                    res += f"{i+1}. {s}\n"
                res += f"\nThis resolves the inquiry regarding '{query}' by detailing these specific points directly from the records."
                return res
                
        # Default mock replies if no context is found
        if "revenue" in q_clean or "financial" in q_clean:
            return "According to the financial reports, the company recorded total revenues of $12.4 Billion for the fiscal year, representing an 8% year-over-year increase, driven primarily by enterprise software subscriptions."
        elif "architecture" in q_clean or "pipeline" in q_clean:
            return "The software architecture leverages a hybrid retrieval engine combining dense vector indices and BM25 sparse matching. The components are fully decoupled, with an API serving layer and a real-time LLM validation pipeline."
        elif "policy" in q_clean or "hr" in q_clean:
            return "The HR policy states that all remote work requests must be submitted through the portal by the 1st of each month. Approvals are processed by direct managers within 5 business days."
            
        return f"This is a simulated offline response from the RAG engine for your question: '{query}'. Please start Ollama and set 'llm.provider: ollama' in config.yaml to get live model completions."

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Public entry point for text completion."""
        if self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt)
        return self._call_mock(prompt, system_prompt)

    def rewrite_query(self, query: str) -> str:
        """
        Query Expansion: Rewrites conversational queries into structured, search-friendly terms.
        """
        system_prompt = (
            "You are an AI Search Specialist. Your task is to rewrite the user's conversational query "
            "into 3-4 keywords or structured search terms that will optimize retrieval from a vector database. "
            "Respond ONLY with the search terms separated by spaces, no formatting, explanation or introductory text."
        )
        prompt = f"User conversational query: '{query}'\nOptimized search query:"
        
        if self.provider == "ollama":
            rewritten = self._call_ollama(prompt, system_prompt)
            # Clean up potential LLM conversational garbage
            rewritten = rewritten.replace('"', '').replace("'", "").strip()
            return rewritten
            
        # Mock rewriting (remove filler words)
        filler_words = {"what", "is", "the", "a", "an", "for", "of", "about", "can", "you", "tell", "me", "show"}
        words = query.lower().split()
        keywords = [w for w in words if w not in filler_words]
        return " ".join(keywords) if keywords else query

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generates a final answer using retrieved context passages.
        """
        context_block = "\n".join([f"--- Context {i+1} ---\n{ctx}" for i, ctx in enumerate(contexts)])
        
        system_prompt = (
            "You are a professional enterprise assistant. Answer the user's question using ONLY the provided contexts. "
            "If the answer cannot be found in the context, say 'I cannot find the answer in the provided documents.' "
            "Do not make up facts or extrapolate beyond what is explicitly stated in the context."
        )
        
        prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        
        return self.generate(prompt, system_prompt)
