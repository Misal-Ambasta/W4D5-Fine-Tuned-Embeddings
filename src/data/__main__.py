import os
import sys
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_processor import DataProcessor

def main():
    """Process the raw data and prepare it for training"""
    print("Processing sales conversation data...")
    
    # Ensure directories exist
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Process the data
    input_path = os.path.join("data", "raw", "sample_sales_data.csv")
    output_path = os.path.join("data", "processed", "processed_sales_data.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        print("Please run 'python generate_sample_data.py' first.")
        return 1
    
    # Process the data
    processor.process_csv(input_path, output_path)
    
    print(f"\nProcessed data saved to {output_path}")
    print("\nNext steps:")
    print("1. Start the application: python run_all.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())