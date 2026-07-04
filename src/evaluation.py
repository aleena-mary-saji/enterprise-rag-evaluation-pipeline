import re
import json
from typing import List, Dict, Any
from src.generation import LLMClient

class RAGEvaluator:
    """
    Implements the 'LLM-as-a-Judge' framework evaluating the RAG Triad:
    1. Faithfulness (Groundedness)
    2. Answer Relevance
    3. Context Precision (MAP-based)
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Args:
            llm_client: Instantiated LLMClient used to run evaluation prompts.
        """
        self.llm_client = llm_client

    def _parse_json_response(self, response_text: str, default_val: Dict[str, Any]) -> Dict[str, Any]:
        """Tolerant JSON parser to extract structured evaluation results from LLM outputs."""
        if not response_text:
            return default_val
            
        # Clean potential markdown block formatting
        cleaned = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        
        # Try direct load
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex search for anything inside { ... }
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
                    
        print(f"[Eval Warning] Failed to parse JSON from response. Raw text: {response_text[:120]}...")
        return default_val

    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Evaluates Faithfulness: Is the answer grounded *only* in the retrieved context?
        """
        if not answer or not contexts:
            return {"score": 0.0, "reasoning": "Answer or Context is empty.", "claims": []}
            
        if self.llm_client.provider == "mock":
            # Simulate a realistic score
            return {
                "score": 0.95 if "provided documentation" in answer or len(contexts) > 0 else 0.5,
                "reasoning": "Mock verification: 9/10 claims verified in the simulated contexts.",
                "claims": [{"claim": "Answer is derived from context", "supported": True}]
            }
            
        context_text = "\n".join(contexts)
        prompt = (
            "Analyze the Candidate Answer and Reference Context provided below. You must perform two tasks:\n"
            "1. Extract all factual statements (claims) made in the Candidate Answer.\n"
            "2. For each claim, check if it is directly supported by the Reference Context.\n\n"
            f"Reference Context:\n{context_text}\n\n"
            f"Candidate Answer:\n{answer}\n\n"
            "Respond ONLY as a valid JSON object matching this schema:\n"
            "{\n"
            "  \"claims\": [\n"
            "    {\"claim\": \"string statement\", \"supported\": true/false}\n"
            "  ],\n"
            "  \"reasoning\": \"string summarizing the assessment\"\n"
            "}"
        )
        
        system_prompt = "You are a strict factual auditor. Respond strictly in JSON. Do not include any explanations outside the JSON."
        raw_response = self.llm_client.generate(prompt, system_prompt)
        parsed = self._parse_json_response(raw_response, {"claims": [], "reasoning": "Failed to parse judge output."})
        
        claims = parsed.get("claims", [])
        if not claims:
            return {"score": 1.0, "reasoning": "No factual claims found in the answer.", "claims": []}
            
        supported_count = sum(1 for c in claims if c.get("supported") is True)
        score = float(supported_count / len(claims))
        
        return {
            "score": score,
            "reasoning": parsed.get("reasoning", f"{supported_count} out of {len(claims)} claims are supported by context."),
            "claims": claims
        }

    def evaluate_answer_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Evaluates Answer Relevance: Does the answer address the user query?
        """
        if not query or not answer:
            return {"score": 0.0, "reasoning": "Query or Answer is empty."}
            
        if self.llm_client.provider == "mock":
            # Simulate
            return {
                "score": 0.9,
                "reasoning": "Mock verification: Answer directly addresses the prompt keywords."
            }
            
        prompt = (
            "Evaluate how relevant and responsive the Candidate Answer is to the User Query.\n"
            "Rate the relevance on a scale of 1 to 5, where:\n"
            "1: Completely irrelevant or off-topic.\n"
            "3: Partially answers the query but lacks detail or misses core concepts.\n"
            "5: Direct, clear, and fully addresses the question.\n\n"
            f"User Query: {query}\n"
            f"Candidate Answer: {answer}\n\n"
            "Respond ONLY as a valid JSON object matching this schema:\n"
            "{\n"
            "  \"rating\": 1-5 integer,\n"
            "  \"reasoning\": \"string summarizing why this score was given\"\n"
            "}"
        )
        
        system_prompt = "You are a helpful UI evaluator. Respond strictly in JSON. Do not include introductory text."
        raw_response = self.llm_client.generate(prompt, system_prompt)
        parsed = self._parse_json_response(raw_response, {"rating": 3, "reasoning": "Failed to parse judge output."})
        
        rating = parsed.get("rating", 3)
        # Normalize rating from 1-5 scale to 0.0 - 1.0
        score = float((rating - 1) / 4)
        
        return {
            "score": max(0.0, min(1.0, score)),
            "reasoning": parsed.get("reasoning", f"Rating: {rating}/5")
        }

    def _evaluate_chunk_relevance(self, query: str, chunk_text: str) -> bool:
        """Helper to determine if a single chunk is relevant to the query."""
        if self.llm_client.provider == "mock":
            # Simple heuristic for mock relevance: keyword overlap
            query_words = set(query.lower().split())
            chunk_words = set(chunk_text.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            return overlap > 0
            
        prompt = (
            "Analyze the User Query and the retrieved document Chunk.\n"
            "Determine if the chunk contains useful or relevant details to help answer the query.\n\n"
            f"User Query: {query}\n"
            f"Chunk Text: {chunk_text}\n\n"
            "Respond ONLY as a valid JSON object matching this schema:\n"
            "{\n"
            "  \"relevant\": true/false,\n"
            "  \"reasoning\": \"brief reasoning\"\n"
            "}"
        )
        
        system_prompt = "You are an expert search relevance judge. Respond strictly in JSON."
        raw_response = self.llm_client.generate(prompt, system_prompt)
        parsed = self._parse_json_response(raw_response, {"relevant": False})
        return bool(parsed.get("relevant", False))

    def evaluate_context_precision(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates Context Precision: Are the retrieved context chunks relevant, 
        and are the highly relevant ones ranked at the top?
        Uses Mean Average Precision (MAP) calculations.
        """
        if not query or not retrieved_chunks:
            return {"score": 0.0, "reasoning": "Query or retrieved chunks list is empty.", "relevance_vector": []}
            
        relevance_vector = []
        reasoning_list = []
        
        # Check relevance for each retrieved chunk
        for i, chunk in enumerate(retrieved_chunks):
            is_relevant = self._evaluate_chunk_relevance(query, chunk['text'])
            relevance_vector.append(is_relevant)
            reasoning_list.append(f"Chunk {i+1}: {'Relevant' if is_relevant else 'Not Relevant'}")
            
        # Calculate Context Precision (AP at each relevant index)
        # Precision@k = (relevant chunks up to k) / k
        relevant_count = 0
        precision_sum = 0.0
        
        for k, is_rel in enumerate(relevance_vector):
            if is_rel:
                relevant_count += 1
                precision_at_k = relevant_count / (k + 1)
                precision_sum += precision_at_k
                
        if relevant_count == 0:
            score = 0.0
        else:
            score = float(precision_sum / len(relevance_vector)) # Average Precision
            
        return {
            "score": score,
            "reasoning": f"Precision calculated over {len(retrieved_chunks)} chunks: {', '.join(reasoning_list)}.",
            "relevance_vector": relevance_vector
        }

    def evaluate_all(self, query: str, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs all three evaluations and returns a combined summary."""
        contexts = [c['text'] for c in retrieved_chunks]
        
        faithfulness = self.evaluate_faithfulness(answer, contexts)
        relevance = self.evaluate_answer_relevance(query, answer)
        precision = self.evaluate_context_precision(query, retrieved_chunks)
        
        return {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "context_precision": precision,
            "average_score": float((faithfulness['score'] + relevance['score'] + precision['score']) / 3)
        }
