import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.embedding_model import EmbeddingModel

class TestEmbeddingModel(unittest.TestCase):
    """Test cases for the EmbeddingModel class"""
    
    def setUp(self):
        """Set up test environment"""
        self.model = EmbeddingModel()
    
    def test_generate_embedding(self):
        """Test embedding generation"""
        text = "This is a test sentence for embedding generation."
        embedding = self.model.generate_embedding(text)
        
        # Check that embedding is a numpy array
        self.assertIsInstance(embedding, np.ndarray)
        
        # Check embedding dimensions (depends on the model)
        self.assertTrue(len(embedding.shape) == 1)  # Should be a 1D array
        
        # For the default model (all-MiniLM-L6-v2), embedding size should be 384
        # If using a different model, this might need adjustment
        self.assertEqual(embedding.shape[0], 384)
    
    def test_predict_conversion(self):
        """Test conversion prediction"""
        # Positive example
        positive_text = "I'm very interested in your product. When can we start?"
        positive_embedding = self.model.generate_embedding(positive_text)
        positive_prediction = self.model.predict_conversion(positive_embedding)
        
        # Negative example
        negative_text = "I'm not sure this is what we need right now."
        negative_embedding = self.model.generate_embedding(negative_text)
        negative_prediction = self.model.predict_conversion(negative_embedding)
        
        # Check that predictions are floats between 0 and 1
        self.assertIsInstance(positive_prediction, float)
        self.assertIsInstance(negative_prediction, float)
        self.assertTrue(0 <= positive_prediction <= 1)
        self.assertTrue(0 <= negative_prediction <= 1)
        
        # Note: Without a trained model, we can't assert specific prediction values
        # This just checks the basic functionality

if __name__ == "__main__":
    unittest.main()