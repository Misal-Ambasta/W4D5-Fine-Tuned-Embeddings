import os
import sys
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from src.evaluation.metrics import ModelEvaluator

def run_evaluation():
    """Run evaluation comparing base and fine-tuned models"""
    print("Running model evaluation...")
    
    # Check if processed data exists
    data_path = os.path.join("data", "processed", "processed_sales_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: Processed data file {data_path} not found.")
        print("Please run 'python -m src.data.data_processor' first.")
        return 1
    
    # Check if fine-tuned model exists
    fine_tuned_model_path = os.path.join("models", "fine_tuned", "fine_tuned_model")
    if not os.path.exists(fine_tuned_model_path):
        print(f"Warning: Fine-tuned model not found at {fine_tuned_model_path}.")
        print("Will only evaluate base model. To train a fine-tuned model, use the API or Streamlit app.")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load data
    print("\nLoading test data...")
    evaluator.load_test_data(data_path)
    
    # Evaluate base model
    print("\nEvaluating base model...")
    base_metrics = evaluator.evaluate_base_model()
    print("\nBase Model Metrics:")
    print(evaluator.format_metrics(base_metrics))
    
    # Evaluate fine-tuned model if it exists
    if os.path.exists(fine_tuned_model_path):
        print("\nEvaluating fine-tuned model...")
        fine_tuned_metrics = evaluator.evaluate_fine_tuned_model()
        print("\nFine-tuned Model Metrics:")
        print(evaluator.format_metrics(fine_tuned_metrics))
        
        # Compare models
        print("\nModel Comparison:")
        comparison = evaluator.compare_models()
        for metric, improvement in comparison.items():
            print(f"{metric}: {improvement:+.2f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(run_evaluation())