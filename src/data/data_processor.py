import os
import pandas as pd
import json
import csv
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing sales conversation data"""
    
    def __init__(self):
        """Initialize the data processor"""
        # Load environment variables
        self.raw_data_dir = os.getenv("RAW_DATA_DIR", "./data/raw")
        self.processed_data_dir = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
        
        # Ensure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def process_transcripts(self, file_path: str) -> str:
        """Process transcript file and prepare for training
        
        Args:
            file_path: Path to the transcript file
            
        Returns:
            Path to the processed file
        """
        logger.info(f"Processing transcript file: {file_path}")
        
        # Determine file type and load data
        if file_path.endswith(".csv"):
            data = self._load_csv(file_path)
        elif file_path.endswith(".json"):
            data = self._load_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Process the data
        processed_data = self._process_data(data)
        
        # Save processed data
        output_path = os.path.join(self.processed_data_dir, "processed_sales_data.csv")
        self._save_csv(processed_data, output_path)
        
        logger.info(f"Processed data saved to {output_path}")
        return output_path
    
    def _load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of dictionaries with data
        """
        try:
            df = pd.read_csv(file_path)
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def _load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of dictionaries with data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            else:
                return [data]
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}")
            raise
    
    def _process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the data for training
        
        Args:
            data: List of dictionaries with data
            
        Returns:
            Processed data
        """
        processed = []
        
        # Check required fields and normalize data structure
        for item in data:
            processed_item = {}
            
            # Extract text field (handle different possible field names)
            text = None
            for field in ["text", "transcript", "conversation", "content"]:
                if field in item and item[field]:
                    text = item[field]
                    break
            
            if not text:
                logger.warning(f"Skipping item without text content: {item}")
                continue
            
            processed_item["text"] = self._clean_text(text)
            
            # Extract conversion outcome (handle different possible field names)
            converted = None
            for field in ["converted", "conversion", "success", "outcome"]:
                if field in item:
                    # Handle different formats (boolean, string, numeric)
                    value = item[field]
                    if isinstance(value, bool):
                        converted = value
                    elif isinstance(value, str):
                        converted = value.lower() in ["true", "yes", "1", "success", "converted"]
                    elif isinstance(value, (int, float)):
                        converted = bool(value)
                    break
            
            if converted is None:
                logger.warning(f"Skipping item without conversion outcome: {item}")
                continue
            
            processed_item["converted"] = converted
            
            # Extract metadata (optional)
            metadata_fields = ["customer_type", "product", "duration", "agent", "date", "time"]
            for field in metadata_fields:
                if field in item:
                    processed_item[field] = item[field]
            
            processed.append(processed_item)
        
        logger.info(f"Processed {len(processed)} items from {len(data)} total")
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (optional, depending on requirements)
        # text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def _save_csv(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Save data to CSV file
        
        Args:
            data: List of dictionaries with data
            output_path: Path to save the CSV file
        """
        try:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        except Exception as e:
            logger.error(f"Error saving CSV file: {str(e)}")
            raise
    
    def generate_sample_data(self, output_path: str, num_samples: int = 100) -> str:
        """Generate sample data for testing
        
        Args:
            output_path: Path to save the sample data
            num_samples: Number of samples to generate
            
        Returns:
            Path to the generated file
        """
        logger.info(f"Generating {num_samples} sample data points")
        
        # Sample positive examples (converted=True)
        positive_examples = [
            "I'm very interested in your product. Can you tell me more about pricing?",
            "This solution addresses exactly what we've been looking for. When can we start?",
            "I see the value in what you're offering. Let's discuss implementation details.",
            "Your product seems to solve our main pain points. I'd like to move forward.",
            "I'm impressed with the features. I think this would work well for our team."
        ]
        
        # Sample negative examples (converted=False)
        negative_examples = [
            "I'm not sure this is what we need right now. Let me think about it.",
            "The price point is higher than we budgeted for. We'll need to reconsider.",
            "I need to discuss this with my team before making any decisions.",
            "We're currently using a competitor's product and are satisfied with it.",
            "This doesn't have all the features we're looking for. We'll keep looking."
        ]
        
        # Generate data
        data = []
        for i in range(num_samples):
            # Determine if this sample is positive or negative
            is_positive = i % 2 == 0
            
            if is_positive:
                base_text = positive_examples[i % len(positive_examples)]
            else:
                base_text = negative_examples[i % len(negative_examples)]
            
            # Add some variation
            text = f"Sample conversation {i+1}: {base_text} Additional context to make each sample unique."
            
            # Create sample
            sample = {
                "text": text,
                "converted": is_positive,
                "customer_type": "new" if i % 3 == 0 else "existing",
                "product": f"Product {(i % 5) + 1}",
                "duration": (i % 30) + 5  # 5-35 minutes
            }
            
            data.append(sample)
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Sample data saved to {output_path}")
        return output_path