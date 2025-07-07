import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Class for generating embeddings and making predictions using fine-tuned models"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the embedding model
        
        Args:
            model_path: Path to the fine-tuned model. If None, uses the default model from env vars
        """
        # Load environment variables
        self.base_model_name = os.getenv("BASE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.fine_tuned_model_path = model_path or os.getenv("FINE_TUNED_MODEL_PATH", "./models/fine_tuned/sales_embeddings")
        
        # Check if fine-tuned model exists and is properly trained, otherwise use base model
        try:
            if os.path.exists(self.fine_tuned_model_path) and os.path.exists(os.path.join(self.fine_tuned_model_path, "config.json")):
                logger.info(f"Loading fine-tuned model from {self.fine_tuned_model_path}")
                self.model = SentenceTransformer(self.fine_tuned_model_path)
                self.using_fine_tuned = True
            else:
                logger.info(f"Fine-tuned model not found or incomplete. Loading base model {self.base_model_name}")
                self.model = SentenceTransformer(self.base_model_name)
                self.using_fine_tuned = False
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}. Falling back to base model.")
            self.model = SentenceTransformer(self.base_model_name)
            self.using_fine_tuned = False
        
        # Load example database if available
        self.examples_db = self._load_examples_db()
    
    def _load_examples_db(self) -> List[Dict[str, Any]]:
        """Load example database for similarity search"""
        examples_path = Path("./data/processed/examples_db.json")
        if examples_path.exists():
            try:
                with open(examples_path, 'r') as f:
                    examples = json.load(f)
                logger.info(f"Loaded {len(examples)} examples from database")
                return examples
            except Exception as e:
                logger.error(f"Error loading examples database: {str(e)}")
                return []
        else:
            logger.warning("Examples database not found")
            return []
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text
        
        Args:
            text: The text to generate embedding for
            
        Returns:
            Embedding vector as numpy array
        """
        # Preprocess text if needed
        processed_text = self._preprocess_text(text)
        
        # Generate embedding
        embedding = self.model.encode(processed_text, convert_to_numpy=True)
        
        return embedding
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before generating embeddings
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Simple preprocessing - can be expanded as needed
        processed = text.strip()
        return processed

    def get_prediction(self, text: str) -> Dict[str, Any]:
        """Generate a full prediction for a given transcript, including embedding and similar examples."""
        embedding = self.generate_embedding(text)
        conversion_probability = self.predict_conversion(embedding)
        similar_examples = self.find_similar_examples(embedding)

        return {
            "conversion_probability": float(conversion_probability),
            "embedding": embedding.tolist(),
            "similar_examples": similar_examples
        }
    
    def predict_conversion(self, embedding: np.ndarray) -> float:
        """Predict conversion probability from embedding
        
        Args:
            embedding: The embedding vector
            
        Returns:
            Conversion probability (0-1)
        """
        # If we have examples, use similarity-based prediction
        if self.examples_db:
            return self._similarity_based_prediction(embedding)
        else:
            # Fallback to a simple heuristic if no examples available
            # This is just a placeholder - in a real system, you'd have a trained classifier
            logger.warning("No examples database available. Using fallback prediction method.")
            return 0.5  # Default 50% probability
    
    def _similarity_based_prediction(self, query_embedding: np.ndarray) -> float:
        """Predict conversion based on similarity to known examples
        
        Args:
            query_embedding: Embedding of the query text
            
        Returns:
            Conversion probability
        """
        if not self.examples_db:
            return 0.5
        
        # Get embeddings for all examples
        example_texts = [example["text"] for example in self.examples_db]
        example_embeddings = self.model.encode(example_texts, convert_to_numpy=True)
        
        # Calculate cosine similarities
        similarities = util.cos_sim(torch.tensor([query_embedding]), 
                                   torch.tensor(example_embeddings))[0].numpy()
        
        # Get conversion labels
        labels = np.array([example["converted"] for example in self.examples_db], dtype=float)
        
        # Weight by similarity
        weights = np.exp(similarities) / np.sum(np.exp(similarities))  # Softmax to get weights
        weighted_prediction = np.sum(weights * labels)
        
        return float(weighted_prediction)
    
    def find_similar_examples(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar examples to the query
        
        Args:
            query_embedding: Embedding of the query text
            top_k: Number of similar examples to return
            
        Returns:
            List of similar examples with similarity scores
        """
        if not self.examples_db:
            return []
        
        # Get embeddings for all examples
        example_texts = [example["text"] for example in self.examples_db]
        example_embeddings = self.model.encode(example_texts, convert_to_numpy=True)
        
        # Calculate cosine similarities
        similarities = util.cos_sim(torch.tensor([query_embedding]), 
                                   torch.tensor(example_embeddings))[0].numpy()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare result
        result = []
        for idx in top_indices:
            example = self.examples_db[idx].copy()
            example["similarity"] = float(similarities[idx])
            result.append(example)
        
        return result
    
    def save_example(self, text: str, converted: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save an example to the database
        
        Args:
            text: The text of the example
            converted: Whether the example resulted in conversion
            metadata: Optional metadata about the example
        """
        # Create example object
        example = {
            "text": text,
            "converted": converted,
            "metadata": metadata or {}
        }
        
        # Add to database
        self.examples_db.append(example)
        
        # Save to disk
        examples_path = Path("./data/processed/examples_db.json")
        try:
            with open(examples_path, 'w') as f:
                json.dump(self.examples_db, f)
            logger.info(f"Saved example to database. Total examples: {len(self.examples_db)}")
        except Exception as e:
            logger.error(f"Error saving example to database: {str(e)}")
    
    def get_base_model(self) -> SentenceTransformer:
        """Get the base model for fine-tuning
        
        Returns:
            Base SentenceTransformer model
        """
        return SentenceTransformer(self.base_model_name)