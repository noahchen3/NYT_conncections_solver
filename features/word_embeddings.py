"""
Load and use pre-trained word embeddings.
"""
import gensim.downloader as api
from typing import Optional
import numpy as np
import pickle
from pathlib import Path

# Global cache for embedding models (loaded once, reused across instances)
_embedding_model_cache = {}

# Directory for cached models
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class WordEmbeddings:
    """Wrapper for pre-trained word embeddings."""
    
    def __init__(self, model_name: str = "glove-wiki-gigaword-300"):
        """
        Initialize word embeddings.
        Uses global cache and disk cache to avoid reloading models.
        
        Args:
            model_name: Name of the embedding model to load
        """
        self.model_name = model_name
        global _embedding_model_cache
        
        # Check if model is already cached in memory
        if model_name in _embedding_model_cache:
            self.model = _embedding_model_cache[model_name]
        else:
            self.model = None
            self._load_model()
            # Cache the model for future use (both in memory and on disk)
            if self.model is not None:
                _embedding_model_cache[model_name] = self.model
    
    def _load_model(self):
        """Load the embedding model from disk cache or gensim API."""
        # Create cache filename from model name (replace hyphens with underscores for filename)
        cache_filename = self.model_name.replace("-", "_") + ".pkl"
        cache_path = MODELS_DIR / cache_filename
        
        # Try to load from disk cache first
        if cache_path.exists():
            try:
                print(f"Loading word embeddings model from cache: {cache_path}...")
                with open(cache_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"✓ Successfully loaded {self.model_name} from cache")
                return
            except Exception as e:
                print(f"Warning: Could not load from cache ({e}), loading from API...")
        
        # Load from gensim API if cache doesn't exist or failed
        print(f"Loading word embeddings model: {self.model_name} (this may take 30-60 seconds on first run)...")
        try:
            self.model = api.load(self.model_name)
            print(f"✓ Successfully loaded {self.model_name}")
            
            # Save to disk cache for future use
            try:
                print(f"Saving model to cache: {cache_path}...")
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.model, f)
                print(f"✓ Successfully cached {self.model_name}")
            except Exception as e:
                print(f"Warning: Could not save to cache ({e}), but model is loaded")
                
        except Exception as e:
            print(f"Warning: Could not load {self.model_name}: {e}")
            print("Falling back to word2vec-google-news-300")
            try:
                self.model = api.load("word2vec-google-news-300")
                print("✓ Successfully loaded word2vec-google-news-300")
                
                # Save fallback model to cache
                fallback_cache_filename = "word2vec_google_news_300.pkl"
                fallback_cache_path = MODELS_DIR / fallback_cache_filename
                try:
                    with open(fallback_cache_path, 'wb') as f:
                        pickle.dump(self.model, f)
                    print(f"✓ Successfully cached word2vec-google-news-300")
                except Exception as e:
                    print(f"Warning: Could not save fallback model to cache ({e})")
                    
            except Exception as e2:
                print(f"Error loading word2vec: {e2}")
                self.model = None
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a word or phrase.
        Handles multi-word phrases by averaging word embeddings.
        
        Args:
            word: Word or phrase to embed
            
        Returns:
            Embedding vector or None if word not found
        """
        if self.model is None:
            return None
        
        # Handle multi-word phrases (e.g., "SOLAR PANEL")
        words = word.split()
        if len(words) > 1:
            # Average embeddings of individual words
            embeddings = []
            for w in words:
                emb = self._get_single_word_embedding(w)
                if emb is not None:
                    embeddings.append(emb)
            
            if embeddings:
                # Average the embeddings
                return np.mean(embeddings, axis=0)
            return None
        
        # Single word - try different case variations
        return self._get_single_word_embedding(word)
    
    def _get_single_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a single word, trying different case variations."""
        word_upper = word.upper()
        word_lower = word.lower()
        word_title = word.title()
        
        # Try different case variations
        for word_variant in [word, word_upper, word_lower, word_title]:
            try:
                return self.model[word_variant]
            except KeyError:
                continue
        
        return None
    
    def __call__(self, word: str) -> np.ndarray:
        """
        Allow WordEmbeddings to be called as a function (for K-Means solver compatibility).
        
        Args:
            word: Word or phrase to embed
            
        Returns:
            Embedding vector (returns zero vector if word not found)
        """
        embedding = self.get_embedding(word)
        if embedding is None:
            # Return zero vector with same dimension as GloVe (300 dimensions)
            return np.zeros(300)
        return embedding
    
    def cosine_similarity(self, word1: str, word2: str) -> float:
        """
        Compute cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between -1 and 1, or 0 if words not found
        """
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Normalize vectors (use max to ensure minimum threshold for division)
        emb1_norm = emb1 / max(np.linalg.norm(emb1), 1e-8)
        emb2_norm = emb2 / max(np.linalg.norm(emb2), 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)

