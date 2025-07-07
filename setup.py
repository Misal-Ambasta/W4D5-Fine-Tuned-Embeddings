import os
import sys
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from src.data.sample_generator import generate_sample_data
from src.data.data_processor import DataProcessor

def setup_project():
    """Set up the project by creating necessary directories and sample data"""
    print("Setting up Fine-Tuned Embeddings for Sales Conversion Prediction project...")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed",
        "models/base",
        "models/fine_tuned",
        "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Generate sample data
    print("Generating sample data...")
    sample_data_path = os.path.join("data", "raw", "sample_sales_data.csv")
    generate_sample_data(sample_data_path, num_samples=100)
    
    # Process sample data
    print("Processing sample data...")
    data_processor = DataProcessor()
    processed_path = data_processor.process_transcripts(sample_data_path)
    
    print(f"\nSetup complete! Sample data has been generated and processed.")
    print(f"Raw data: {sample_data_path}")
    print(f"Processed data: {processed_path}")
    print("\nNext steps:")
    print("1. Start the FastAPI backend: uvicorn api.main:app --reload")
    print("2. Start the Streamlit frontend: streamlit run app/main.py")

if __name__ == "__main__":
    setup_project()