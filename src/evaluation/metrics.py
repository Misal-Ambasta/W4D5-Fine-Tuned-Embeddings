import os
import numpy as np
import pandas as pd
import json
import torch
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Class for evaluating and comparing embedding models"""
    
    def __init__(self):
        """Initialize the model evaluator"""
        # Load environment variables
        self.base_model_name = os.getenv("BASE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.fine_tuned_model_path = os.getenv("FINE_TUNED_MODEL_PATH", "./models/fine_tuned/sales_embeddings")
        self.processed_data_dir = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
        
        # Load models if available
        self.fine_tuned_model = self._load_fine_tuned_model()
        self.base_model = self._load_base_model()
        
        # Cache for evaluation results
        self._metrics_cache = None
        self._comparison_cache = None
    
    def _load_fine_tuned_model(self) -> Optional[SentenceTransformer]:
        """Load the fine-tuned model if available"""
        if os.path.exists(self.fine_tuned_model_path):
            try:
                logger.info(f"Loading fine-tuned model from {self.fine_tuned_model_path}")
                return SentenceTransformer(self.fine_tuned_model_path)
            except Exception as e:
                logger.error(f"Error loading fine-tuned model: {str(e)}")
                return None
        else:
            logger.warning(f"Fine-tuned model not found at {self.fine_tuned_model_path}")
            return None
    
    def _load_base_model(self) -> SentenceTransformer:
        """Load the base model"""
        try:
            logger.info(f"Loading base model {self.base_model_name}")
            return SentenceTransformer(self.base_model_name)
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise
    
    def get_metrics(self, force_recalculate: bool = False) -> Dict[str, Any]:
        """Get model performance metrics
        
        Args:
            force_recalculate: Whether to force recalculation of metrics
            
        Returns:
            Dictionary with metrics
        """
        # Return cached metrics if available and not forcing recalculation
        if self._metrics_cache is not None and not force_recalculate:
            return self._metrics_cache
        
        # Check if fine-tuned model is available
        if self.fine_tuned_model is None:
            return self._get_dummy_metrics()
        
        # Load test data
        test_data = self._load_test_data()
        if not test_data:
            return self._get_dummy_metrics()
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_data)
        
        # Cache the results
        self._metrics_cache = metrics
        
        return metrics
    
    def compare_models(self, force_recalculate: bool = False) -> Dict[str, Any]:
        """Compare fine-tuned vs generic embeddings
        
        Args:
            force_recalculate: Whether to force recalculation of comparison
            
        Returns:
            Dictionary with comparison results
        """
        # Return cached comparison if available and not forcing recalculation
        if self._comparison_cache is not None and not force_recalculate:
            return self._comparison_cache
        
        # Check if both models are available
        if self.fine_tuned_model is None or self.base_model is None:
            return self._get_dummy_comparison()
        
        # Load test data
        test_data = self._load_test_data()
        if not test_data:
            return self._get_dummy_comparison()
        
        # Calculate metrics for both models
        fine_tuned_metrics = self._calculate_metrics(test_data, model=self.fine_tuned_model)
        base_metrics = self._calculate_metrics(test_data, model=self.base_model)
        
        # Prepare comparison
        comparison = {
            "fine_tuned": fine_tuned_metrics,
            "generic": base_metrics,
            "improvement": {}
        }
        
        # Calculate improvement
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
            if metric in fine_tuned_metrics and metric in base_metrics:
                fine_tuned_value = fine_tuned_metrics[metric]
                base_value = base_metrics[metric]
                improvement = fine_tuned_value - base_value
                improvement_percent = (improvement / base_value) * 100 if base_value > 0 else 0
                
                comparison["improvement"][metric] = {
                    "absolute": improvement,
                    "percent": improvement_percent
                }
        
        # Cache the results
        self._comparison_cache = comparison
        
        return comparison
    
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data for evaluation
        
        Returns:
            List of test data items
        """
        # Try to load examples database
        examples_path = os.path.join(self.processed_data_dir, "examples_db.json")
        if os.path.exists(examples_path):
            try:
                with open(examples_path, 'r') as f:
                    examples = json.load(f)
                
                # Split into train/test if needed
                if len(examples) > 10:  # Only split if we have enough examples
                    _, test_examples = train_test_split(examples, test_size=0.2, random_state=42)
                    return test_examples
                else:
                    return examples
            except Exception as e:
                logger.error(f"Error loading examples database: {str(e)}")
        
        # If no examples database, try to load processed data
        processed_data_path = os.path.join(self.processed_data_dir, "processed_sales_data.csv")
        if os.path.exists(processed_data_path):
            try:
                df = pd.read_csv(processed_data_path)
                test_df = df.sample(frac=0.2, random_state=42)
                return test_df.to_dict(orient="records")
            except Exception as e:
                logger.error(f"Error loading processed data: {str(e)}")
        
        logger.warning("No test data found")
        return []
    
    def _calculate_metrics(self, test_data: List[Dict[str, Any]], 
                          model: Optional[SentenceTransformer] = None) -> Dict[str, float]:
        """Calculate performance metrics
        
        Args:
            test_data: List of test data items
            model: Model to use for evaluation (defaults to fine-tuned model)
            
        Returns:
            Dictionary with metrics
        """
        if model is None:
            model = self.fine_tuned_model
            if model is None:
                return self._get_dummy_metrics()
        
        # Extract texts and labels
        texts = [item["text"] for item in test_data]
        true_labels = np.array([int(item["converted"]) for item in test_data])
        
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        # Calculate similarity matrix
        similarity_matrix = util.cos_sim(torch.tensor(embeddings), torch.tensor(embeddings)).numpy()
        
        # Predict labels using similarity-based approach
        predicted_probs = np.zeros(len(test_data))
        for i in range(len(test_data)):
            # Get similarities to other examples (excluding self)
            similarities = similarity_matrix[i].copy()
            similarities[i] = 0  # Exclude self-similarity
            
            # Get top-k similar examples
            k = min(5, len(test_data) - 1)
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Weight by similarity
            weights = similarities[top_indices]
            weights = np.exp(weights) / np.sum(np.exp(weights))  # Softmax to get weights
            
            # Get labels of similar examples
            similar_labels = true_labels[top_indices]
            
            # Calculate weighted prediction
            predicted_probs[i] = np.sum(weights * similar_labels)
        
        # Convert probabilities to binary predictions
        predicted_labels = (predicted_probs >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "precision": precision_score(true_labels, predicted_labels, zero_division=0),
            "recall": recall_score(true_labels, predicted_labels, zero_division=0),
            "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
            "auc_roc": roc_auc_score(true_labels, predicted_probs) if len(np.unique(true_labels)) > 1 else 0.5
        }
        
        return metrics
    
    def _get_dummy_metrics(self) -> Dict[str, float]:
        """Get dummy metrics when no model or data is available
        
        Returns:
            Dictionary with dummy metrics
        """
        return {
            "accuracy": 0.75,
            "precision": 0.70,
            "recall": 0.80,
            "f1_score": 0.75,
            "auc_roc": 0.78,
            "note": "These are placeholder metrics. Train a model for actual performance."
        }
    
    def _get_dummy_comparison(self) -> Dict[str, Any]:
        """Get dummy comparison when models or data are not available
        
        Returns:
            Dictionary with dummy comparison
        """
        fine_tuned = self._get_dummy_metrics()
        generic = {
            "accuracy": 0.65,
            "precision": 0.60,
            "recall": 0.70,
            "f1_score": 0.65,
            "auc_roc": 0.68,
            "note": "These are placeholder metrics. Train a model for actual performance."
        }
        
        improvement = {}
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
            improvement[metric] = {
                "absolute": fine_tuned[metric] - generic[metric],
                "percent": ((fine_tuned[metric] - generic[metric]) / generic[metric]) * 100
            }
        
        return {
            "fine_tuned": fine_tuned,
            "generic": generic,
            "improvement": improvement,
            "note": "These are placeholder comparisons. Train a model for actual performance."
        }