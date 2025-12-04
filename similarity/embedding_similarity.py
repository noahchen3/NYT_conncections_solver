"""
Compute word similarity using embeddings.
"""
from features.word_embeddings import WordEmbeddings
import config


class EmbeddingSimilarity:
    """Computes similarity between words using embeddings."""
    
    def __init__(self):
        """
        Initialize similarity components.
        Always loads models fresh to prevent data leakage.
        """
        # Always create new instance (loads models fresh)
        self.word_embeddings = WordEmbeddings(config.EMBEDDING_MODEL)
    
    def similarity(self, word1: str, word2: str) -> float:
        """
        Compute similarity between two words using embeddings.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize cosine similarity from [-1, 1] to [0, 1]
        embedding_sim = self.word_embeddings.cosine_similarity(word1, word2)
        return self._normalize_similarity(embedding_sim)
    
    def _normalize_similarity(self, sim: float) -> float:
        """Normalize similarity from [-1, 1] to [0, 1]."""
        return (sim + 1.0) / 2.0

