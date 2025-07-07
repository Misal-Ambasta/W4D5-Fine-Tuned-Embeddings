import os
import sys
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from src.data.sample_generator import generate_sample_data

def main():
    """Generate sample data for testing"""
    print("Generating sample sales conversation data...")
    
    # Ensure directory exists
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    
    # Generate sample data
    output_path = os.path.join("data", "raw", "sample_sales_data.csv")
    num_samples = 100
    
    # Allow command line override of number of samples
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of samples: {sys.argv[1]}. Using default: {num_samples}")
    
    generate_sample_data(output_path, num_samples=num_samples)
    
    print(f"\nGenerated {num_samples} sample data points at {output_path}")
    print("\nNext steps:")
    print("1. Process the data: python -m src.data.data_processor")
    print("2. Start the application: python run_all.py")

if __name__ == "__main__":
    main()