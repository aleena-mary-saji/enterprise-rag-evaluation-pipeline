import numpy as np
from src.ingestion import SemanticChunker, RecursiveCharacterChunker

def test_sentence_splitting():
    """Verifies regex sentence splitting is accurate."""
    def mock_embed(texts):
        return [np.zeros(10) for _ in texts]
        
    chunker = SemanticChunker(embed_fn=mock_embed)
    text = "Hello world. This is an audit! Does it work? Yes."
    sentences = chunker.split_into_sentences(text)
    
    assert len(sentences) == 4
    assert sentences[0] == "Hello world."
    assert sentences[1] == "This is an audit!"
    assert sentences[2] == "Does it work?"
    assert sentences[3] == "Yes."

def test_semantic_chunker_logic():
    """
    Tests semantic chunk boundaries by supplying orthogonal vectors for 
    different topics to verify a boundary split occurs.
    """
    def mock_embed(texts):
        # Return orthogonal vectors based on content keywords
        vectors = []
        for t in texts:
            if "revenue" in t:
                vectors.append(np.array([1.0, 0.0]))
            else:
                vectors.append(np.array([0.0, 1.0]))
        return vectors

    chunker = SemanticChunker(
        embed_fn=mock_embed,
        min_chunk_size=10,
        max_chunk_size=1000,
        similarity_threshold=0.5
    )
    
    # Sentence 0 & 1 share "revenue" -> similarity = 1.0 (no split)
    # Sentence 2 contains "architecture" -> similarity to S1 = 0.0 (split)
    text = "The quarterly revenue grew. Our software revenues rose. The systems architecture is modular."
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 2
    assert "quarterly revenue" in chunks[0]["text"]
    assert "software revenues" in chunks[0]["text"]
    assert "systems architecture" in chunks[1]["text"]
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[1]["metadata"]["chunk_index"] == 1

def test_recursive_character_chunker():
    """Verifies recursive chunking respects chunk limits and splits safely."""
    chunker = RecursiveCharacterChunker(chunk_size=30, chunk_overlap=5)
    text = "This is a short sentence. And here is another sentence that is longer."
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c["text"]) <= 30
        assert c["metadata"]["chunk_method"] == "recursive"
