import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FineTuner:
    """Class for fine-tuning embedding models on sales conversation data"""
    
    def __init__(self):
        """Initialize the fine-tuner"""
        # Load environment variables
        self.base_model_name = os.getenv("BASE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.fine_tuned_model_path = os.getenv("FINE_TUNED_MODEL_PATH", "./models/fine_tuned/sales_embeddings")
        self.data_dir = os.getenv("DATA_DIR", "./data")
        self.processed_data_dir = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
        
        # Ensure directories exist
        os.makedirs(self.fine_tuned_model_path, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def train(self, dataset_path: str, epochs: int = 10, batch_size: int = 32, 
              learning_rate: float = 2e-5, max_seq_length: int = 512, 
              eval_split: float = 0.2) -> Dict[str, Any]:
        """Fine-tune the embedding model on sales conversation data
        
        Args:
            dataset_path: Path to the dataset file (CSV or JSON)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            max_seq_length: Maximum sequence length for tokenizer
            eval_split: Fraction of data to use for evaluation
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting fine-tuning with {epochs} epochs, batch size {batch_size}")
        
        # Load and prepare data
        train_examples, eval_examples = self._prepare_data(dataset_path, eval_split)
        
        if not train_examples:
            logger.error("No training examples found")
            return {"error": "No training examples found"}
        
        # Load base model
        model = SentenceTransformer(self.base_model_name)
        model.max_seq_length = max_seq_length
        
        # Prepare training dataloader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Use contrastive loss
        train_loss = losses.ContrastiveLoss(model=model)
        
        # Prepare evaluator
        evaluator = None
        if eval_examples:
            logger.info(f"Using {len(eval_examples)} examples for evaluation")
            evaluator = self._create_evaluator(eval_examples)
        
        # Train the model
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                 epochs=epochs,
                 evaluator=evaluator,
                 evaluation_steps=len(train_dataloader),
                 warmup_steps=int(len(train_dataloader) * 0.1),
                 output_path=self.fine_tuned_model_path,
                 optimizer_params={'lr': learning_rate},
                 show_progress_bar=True)
        
        # Save training metadata
        metadata = {
            "base_model": self.base_model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples) if eval_examples else 0,
            "training_completed": True
        }
        
        with open(os.path.join(self.fine_tuned_model_path, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Fine-tuning completed. Model saved to {self.fine_tuned_model_path}")
        
        return {"status": "success", "model_path": self.fine_tuned_model_path, **metadata}
    
    def _prepare_data(self, dataset_path: str, eval_split: float = 0.2) -> Tuple[List[InputExample], List[InputExample]]:
        """Prepare data for fine-tuning
        
        Args:
            dataset_path: Path to the dataset file
            eval_split: Fraction of data to use for evaluation
            
        Returns:
            Tuple of (train_examples, eval_examples)
        """
        # Load dataset
        try:
            if dataset_path.endswith(".csv"):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith(".json"):
                df = pd.read_json(dataset_path)
            else:
                logger.error(f"Unsupported file format: {dataset_path}")
                return [], []
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return [], []
        
        # Check required columns
        required_columns = ["text", "converted"]
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Dataset missing required columns: {required_columns}")
            return [], []
        
        logger.info(f"Loaded dataset with {len(df)} examples")
        
        # Split into converted and non-converted
        converted_df = df[df["converted"] == True]
        non_converted_df = df[df["converted"] == False]
        
        logger.info(f"Dataset has {len(converted_df)} converted and {len(non_converted_df)} non-converted examples")
        
        # Create pairs for contrastive learning
        examples = self._create_contrastive_pairs(converted_df, non_converted_df)
        
        # Split into train and eval
        train_examples, eval_examples = train_test_split(examples, test_size=eval_split, random_state=42)
        
        logger.info(f"Created {len(train_examples)} training examples and {len(eval_examples)} evaluation examples")
        
        # Also save examples database for similarity search
        self._save_examples_db(df)
        
        return train_examples, eval_examples
    
    def _create_contrastive_pairs(self, converted_df: pd.DataFrame, 
                                 non_converted_df: pd.DataFrame) -> List[InputExample]:
        """Create contrastive pairs for training
        
        Args:
            converted_df: DataFrame with converted examples
            non_converted_df: DataFrame with non-converted examples
            
        Returns:
            List of InputExample objects for training
        """
        examples = []
        
        # Create positive pairs (similar: both converted or both non-converted)
        # Converted pairs
        converted_texts = converted_df["text"].tolist()
        for i in range(len(converted_texts)):
            for j in range(i+1, min(i+5, len(converted_texts))):
                examples.append(InputExample(texts=[converted_texts[i], converted_texts[j]], label=1.0))
        
        # Non-converted pairs
        non_converted_texts = non_converted_df["text"].tolist()
        for i in range(len(non_converted_texts)):
            for j in range(i+1, min(i+5, len(non_converted_texts))):
                examples.append(InputExample(texts=[non_converted_texts[i], non_converted_texts[j]], label=1.0))
        
        # Create negative pairs (dissimilar: one converted, one non-converted)
        for i in range(min(len(converted_texts), 100)):
            for j in range(min(len(non_converted_texts), 5)):
                examples.append(InputExample(texts=[converted_texts[i], non_converted_texts[j]], label=0.0))
        
        return examples
    
    def _create_evaluator(self, eval_examples: List[InputExample]) -> EmbeddingSimilarityEvaluator:
        """Create an evaluator for the model
        
        Args:
            eval_examples: List of evaluation examples
            
        Returns:
            EmbeddingSimilarityEvaluator
        """
        return EmbeddingSimilarityEvaluator.from_input_examples(eval_examples, name='eval')
    
    def _save_examples_db(self, df: pd.DataFrame) -> None:
        """Save examples database for similarity search
        
        Args:
            df: DataFrame with examples
        """
        examples = []
        for _, row in df.iterrows():
            example = {
                "text": row["text"],
                "converted": bool(row["converted"])
            }
            
            # Add any additional metadata columns
            metadata = {}
            for col in df.columns:
                if col not in ["text", "converted"]:
                    metadata[col] = row[col]
            
            if metadata:
                example["metadata"] = metadata
            
            examples.append(example)
        
        # Save to disk
        examples_path = os.path.join(self.processed_data_dir, "examples_db.json")
        try:
            with open(examples_path, 'w') as f:
                json.dump(examples, f, indent=2)
            logger.info(f"Saved {len(examples)} examples to database")
        except Exception as e:
            logger.error(f"Error saving examples to database: {str(e)}")