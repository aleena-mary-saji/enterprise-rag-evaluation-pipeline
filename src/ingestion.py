import os
import re
import numpy as np
from typing import List, Dict, Any

class DocumentLoader:
    """Helper class to load text from TXT, MD, and PDF files."""
    
    @staticmethod
    def load(file_path: str) -> str:
        """Loads the content of a file based on its extension."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            return DocumentLoader._load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
    @staticmethod
    def _load_pdf(file_path: str) -> str:
        """Loads and extracts text from a PDF file using pypdf if available."""
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except ImportError:
            # Fallback warning if pypdf is not installed
            print("[Warning] pypdf not installed. Please install it to parse PDF files. Reading as empty text.")
            return ""


class SemanticChunker:
    """
    Advanced semantic chunker that groups sentences by semantic similarity
    rather than arbitrary character counts.
    """
    
    def __init__(self, embed_fn, min_chunk_size: int = 100, max_chunk_size: int = 800, similarity_threshold: float = 0.65):
        """
        Args:
            embed_fn: A function that takes a list of strings and returns a list/array of embeddings.
            min_chunk_size: Minimum characters per chunk.
            max_chunk_size: Maximum characters per chunk (hard limit).
            similarity_threshold: Cosine similarity threshold below which a split is triggered.
        """
        self.embed_fn = embed_fn
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def split_into_sentences(self, text: str) -> List[str]:
        """Splits raw text into sentences using simple regex rules."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Split by punctuation followed by space and capital letter
        sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes the cosine similarity between two vectors."""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def chunk_text(self, text: str, source_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Splits text into chunks semantically.
        
        Returns:
            List of dictionaries containing 'text' and 'metadata'.
        """
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []
            
        if len(sentences) == 1:
            return [{"text": sentences[0], "metadata": {**(source_metadata or {}), "chunk_index": 0, "chunk_method": "semantic"}}]

        # Generate embeddings for all sentences
        embeddings = self.embed_fn(sentences)
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_len = len(sentences[0])
        chunk_idx = 0
        
        for i in range(1, len(sentences)):
            curr_sentence = sentences[i]
            curr_len = len(curr_sentence)
            
            # Compute similarity between current sentence and previous sentence
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
            # Decide whether to split
            # Split conditions:
            # 1. Similarity falls below threshold AND current chunk has reached min size
            # 2. Adding the current sentence would exceed max chunk size
            should_split = False
            if similarity < self.similarity_threshold and current_chunk_len >= self.min_chunk_size:
                should_split = True
            if current_chunk_len + curr_len > self.max_chunk_size and current_chunk_len >= self.min_chunk_size:
                should_split = True
                
            if should_split:
                # Save the completed chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **(source_metadata or {}),
                        "chunk_index": chunk_idx,
                        "chunk_method": "semantic",
                        "char_length": len(chunk_text),
                        "sentence_count": len(current_chunk_sentences)
                    }
                })
                chunk_idx += 1
                current_chunk_sentences = [curr_sentence]
                current_chunk_len = curr_len
            else:
                current_chunk_sentences.append(curr_sentence)
                current_chunk_len += curr_len + 1 # +1 for the space join
                
        # Append the final remaining chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **(source_metadata or {}),
                    "chunk_index": chunk_idx,
                    "chunk_method": "semantic",
                    "char_length": len(chunk_text),
                    "sentence_count": len(current_chunk_sentences)
                }
            })
            
        return chunks


class RecursiveCharacterChunker:
    """
    Standard hierarchical character-based chunker. Used as a baseline.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not text:
            return []
            
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            # If not at the end, try to find a space to split clean
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space != -1 and last_space > self.chunk_size * 0.7:
                    end = start + last_space
                    chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {
                    **(source_metadata or {}),
                    "chunk_index": chunk_idx,
                    "chunk_method": "recursive",
                    "char_length": len(chunk_text.strip())
                }
            })
            chunk_idx += 1
            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break
                
        return chunks
