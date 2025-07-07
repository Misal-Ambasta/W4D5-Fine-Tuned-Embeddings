import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import torch
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_env_vars(env_file: str = ".env") -> Dict[str, str]:
    """Load environment variables from .env file
    
    Args:
        env_file: Path to the .env file
        
    Returns:
        Dictionary with environment variables
    """
    env_vars = {}
    
    if not os.path.exists(env_file):
        logger.warning(f".env file not found at {env_file}")
        return env_vars
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
                
                # Also set as environment variable
                os.environ[key.strip()] = value.strip()
    except Exception as e:
        logger.error(f"Error loading .env file: {str(e)}")
    
    return env_vars

def ensure_directories_exist(directories: List[str]) -> None:
    """Ensure that directories exist, creating them if necessary
    
    Args:
        directories: List of directory paths
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: Indentation level for JSON formatting
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.debug(f"Saved JSON data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON data to {file_path}: {str(e)}")
        raise

def load_json(file_path: str) -> Any:
    """Load data from JSON file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON data from {file_path}: {str(e)}")
        raise

def cosine_similarity(a: Union[List[float], np.ndarray], 
                      b: Union[List[float], np.ndarray]) -> float:
    """Calculate cosine similarity between two vectors
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (0-1)
    """
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_timestamp() -> str:
    """Get current timestamp as string
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_experiment_dir(base_dir: str = "./models/experiments") -> str:
    """Create a directory for a new experiment
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to the new experiment directory
    """
    timestamp = get_timestamp()
    experiment_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Created experiment directory: {experiment_dir}")
    return experiment_dir

def save_experiment_config(config: Dict[str, Any], experiment_dir: str) -> str:
    """Save experiment configuration
    
    Args:
        config: Experiment configuration
        experiment_dir: Experiment directory
        
    Returns:
        Path to the saved config file
    """
    config_path = os.path.join(experiment_dir, "config.json")
    save_json(config, config_path)
    return config_path

def load_experiment_config(experiment_dir: str) -> Dict[str, Any]:
    """Load experiment configuration
    
    Args:
        experiment_dir: Experiment directory
        
    Returns:
        Experiment configuration
    """
    config_path = os.path.join(experiment_dir, "config.json")
    return load_json(config_path)

def get_device() -> torch.device:
    """Get PyTorch device (CPU or GPU)
    
    Returns:
        PyTorch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """Format metrics for display
    
    Args:
        metrics: Dictionary with metric values
        
    Returns:
        Dictionary with formatted metric values
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.2%}"
        else:
            formatted[key] = str(value)
    return formatted