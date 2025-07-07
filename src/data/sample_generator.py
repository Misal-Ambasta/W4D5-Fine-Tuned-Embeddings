import os
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
import random
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(output_path: str, num_samples: int = 100) -> str:
    """Generate sample sales conversation data for testing
    
    Args:
        output_path: Path to save the sample data
        num_samples: Number of samples to generate
        
    Returns:
        Path to the generated file
    """
    logger.info(f"Generating {num_samples} sample data points")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample positive examples (converted=True)
    positive_examples = [
        "I'm very interested in your product. Can you tell me more about pricing?",
        "This solution addresses exactly what we've been looking for. When can we start?",
        "I see the value in what you're offering. Let's discuss implementation details.",
        "Your product seems to solve our main pain points. I'd like to move forward.",
        "I'm impressed with the features. I think this would work well for our team.",
        "The ROI on this looks promising. I'd like to get started as soon as possible.",
        "This is exactly what our team needs. Let's schedule a demo for the rest of the department.",
        "I can see how this would save us time and money. What's the implementation timeline?",
        "Your solution addresses all our requirements. Let's talk about contract terms.",
        "I'm convinced this is the right choice for us. How soon can we deploy?"
    ]
    
    # Sample negative examples (converted=False)
    negative_examples = [
        "I'm not sure this is what we need right now. Let me think about it.",
        "The price point is higher than we budgeted for. We'll need to reconsider.",
        "I need to discuss this with my team before making any decisions.",
        "We're currently using a competitor's product and are satisfied with it.",
        "This doesn't have all the features we're looking for. We'll keep looking.",
        "I appreciate the presentation, but I don't think it's a good fit for us.",
        "We're not ready to make a change at this time. Maybe next quarter.",
        "The implementation seems too complex for our current resources.",
        "I have concerns about the learning curve for our team.",
        "We need to prioritize other projects right now. Let's reconnect later."
    ]
    
    # Sample customer types
    customer_types = ["New", "Existing", "Returning"]
    
    # Sample product categories
    product_categories = ["Software", "Hardware", "Services", "Consulting", "Training"]
    
    # Generate data
    data = []
    for i in range(num_samples):
        # Determine if this sample is positive or negative
        is_positive = random.random() > 0.5
        
        if is_positive:
            base_text = random.choice(positive_examples)
            # Add some positive-specific phrases
            positive_phrases = [
                "I'm excited about this opportunity.",
                "This looks like a great solution for us.",
                "I can see the value proposition clearly.",
                "Your pricing is within our budget.",
                "The features align well with our needs."
            ]
            additional_text = random.choice(positive_phrases)
        else:
            base_text = random.choice(negative_examples)
            # Add some negative-specific phrases
            negative_phrases = [
                "I have some concerns about the cost.",
                "I'm not seeing how this fits our workflow.",
                "We need to evaluate other options as well.",
                "The timeline doesn't work for us.",
                "I'm not convinced about the ROI."
            ]
            additional_text = random.choice(negative_phrases)
        
        # Create conversation with some structure
        agent_intro = "Agent: Thank you for taking the time to discuss our solution today. I'd like to understand your current challenges and how we might help."
        customer_response = f"Customer: We've been having issues with {random.choice(['efficiency', 'costs', 'quality', 'time management', 'integration'])}."
        agent_pitch = f"Agent: I understand. Our {random.choice(product_categories)} solution is designed specifically to address those challenges by {random.choice(['streamlining processes', 'reducing overhead', 'improving quality', 'saving time', 'seamless integration'])}."
        customer_question = f'''Customer: {random.choice(['How much does it cost?', "What's the implementation time?", 'Do you have case studies?', 'How is this better than competitors?', 'What kind of support do you offer?'])}'''
        agent_answer = "Agent: That's a great question. [Detailed answer with specific information about the product/service]"
        customer_final = f"Customer: {base_text} {additional_text}"
        
        conversation = f"{agent_intro}\n\n{customer_response}\n\n{agent_pitch}\n\n{customer_question}\n\n{agent_answer}\n\n{customer_final}"
        
        # Create sample
        sample = {
            "text": conversation,
            "converted": is_positive,
            "customer_type": random.choice(customer_types),
            "product_category": random.choice(product_categories),
            "call_duration": random.randint(5, 45),  # 5-45 minutes
            "previous_interactions": random.randint(0, 5),
            "agent_id": f"A{random.randint(1000, 9999)}",
            "timestamp": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        }
        
        data.append(sample)
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample data saved to {output_path}")
    return output_path

def main():
    """Main function to generate sample data"""
    output_path = os.path.join("data", "raw", "sample_sales_data.csv")
    generate_sample_data(output_path, num_samples=100)

if __name__ == "__main__":
    main()