import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Constants
# Determine API URL dynamically so frontend works with a custom backend port
_api_port = os.getenv("API_PORT", "8000")
_api_url_override = os.getenv("API_URL")
API_URL = _api_url_override or f"http://localhost:{_api_port}"

# Page configuration
st.set_page_config(
    page_title="Sales Conversion Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1E88E5;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #424242;
    margin-bottom: 1rem;
}
.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.success-box {
    background-color: #d7f9e9;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.warning-box {
    background-color: #fff8e1;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def predict_conversion(text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call the API to predict conversion probability"""
    try:
        response = requests.post(
            f"{API_URL}/api/predict",
            json={"text": text, "metadata": metadata or {}}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error calling prediction API: {str(e)}")
        return {}

def get_metrics() -> Dict[str, Any]:
    """Get model performance metrics"""
    try:
        response = requests.get(f"{API_URL}/api/metrics")
        if response.status_code == 200 and response.text.strip():
            return response.json()
        else:
            return {}
    except Exception as e:
        st.error(f"Error fetching metrics: {str(e)}")
        return {}

def compare_models() -> Dict[str, Any]:
    """Compare fine-tuned vs generic embeddings"""
    try:
        response = requests.get(f"{API_URL}/api/compare")
        return response.json()
    except Exception as e:
        st.error(f"Error comparing models: {str(e)}")
        return {}

def train_model(dataset_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger model training"""
    try:
        response = requests.post(
            f"{API_URL}/api/train",
            json={"dataset_path": dataset_path, **params}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error starting training: {str(e)}")
        return {}

def upload_transcripts(file):
    """Upload transcript file to the API"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_URL}/api/upload/transcripts", files=files)
        return response.json()
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return {}

# Sidebar navigation
st.sidebar.markdown('<p class="main-header">Navigation</p>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Select a page",
    ["Home", "Prediction", "Training", "Evaluation", "About"], key="nav"
)

# Home page
if page == "Home":
    st.markdown('<p class="main-header">Sales Conversion Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Fine-Tuned Embeddings for Sales Conversations</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to the Sales Conversion Prediction System</h3>
    <p>This application uses fine-tuned embeddings to predict the likelihood of sales conversion from call transcripts.</p>
    <p>Navigate through the sidebar to access different features:</p>
    <ul>
        <li><strong>Prediction:</strong> Get conversion predictions for sales transcripts</li>
        <li><strong>Training:</strong> Fine-tune the embedding model on your data</li>
        <li><strong>Evaluation:</strong> View model performance metrics and comparisons</li>
        <li><strong>About:</strong> Learn more about the system</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Display key features
    st.markdown('<p class="sub-header">Key Features</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>Domain-Specific Fine-Tuning</h3>
        <p>Fine-tune pre-trained embeddings on sales conversation data with conversion labels to capture sales-specific semantic patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>Contrastive Learning</h3>
        <p>Train embeddings to distinguish between high-conversion and low-conversion conversation patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
        <h3>LangChain Integration</h3>
        <p>Orchestrate the fine-tuning pipeline, embedding generation, and similarity-based prediction workflow.</p>
        </div>
        """, unsafe_allow_html=True)

# Prediction page
elif page == "Prediction":
    st.markdown('<p class="main-header">Sales Conversion Prediction</p>', unsafe_allow_html=True)
    
    # Input form
    with st.form("prediction_form"):
        # Sample data loader
        sample_text = "Hello John, thank you for taking the time today. I wanted to walk you through ..."
        if st.form_submit_button("Load Sample Transcript"):
            st.session_state["transcript_input"] = sample_text
        transcript = st.text_area("Enter sales call transcript", height=300, key="transcript_input")
        
        # Optional metadata
        st.markdown("### Optional Metadata")
        col1, col2 = st.columns(2)
        with col1:
            customer_type = st.selectbox("Customer Type", ["New", "Existing", "Returning"])
            product_category = st.selectbox("Product Category", ["Software", "Hardware", "Services", "Consulting"])
        with col2:
            call_duration = st.number_input("Call Duration (minutes)", min_value=1, max_value=120, value=15)
            previous_interactions = st.number_input("Previous Interactions", min_value=0, max_value=50, value=0)
        
        metadata = {
            "customer_type": customer_type,
            "product_category": product_category,
            "call_duration": call_duration,
            "previous_interactions": previous_interactions
        }
        
        submitted = st.form_submit_button("Predict Conversion")
    
    if submitted and transcript:
        with st.spinner("Generating prediction..."):
            result = predict_conversion(transcript, metadata)
            
            if "conversion_probability" in result:
                # Display prediction
                prob = result["conversion_probability"]
                st.markdown(f"<div class='success-box'><h2>Conversion Probability: {prob:.2%}</h2></div>", unsafe_allow_html=True)
                
                # Gauge chart for probability
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh([0], [prob], color='#1E88E5', height=0.3)
                ax.barh([0], [1-prob], left=[prob], color='#E0E0E0', height=0.3)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_yticks([])
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                ax.axvline(x=0.5, color='#FFA000', linestyle='--', alpha=0.7)
                st.pyplot(fig)
                
                # Similar examples
                if "similar_examples" in result and result["similar_examples"]:
                    st.markdown("### Similar Conversations")
                    for i, example in enumerate(result["similar_examples"]):
                        with st.expander(f"Example {i+1} - Similarity: {example['similarity']:.2%}"):
                            st.write(example["text"])
                            st.write(f"**Outcome:** {'Converted' if example['converted'] else 'Not Converted'}")
            else:
                st.error("Error generating prediction. Please try again.")

# Training page
elif page == "Training":
    st.markdown('<p class="main-header">Model Training</p>', unsafe_allow_html=True)
    
    # File upload
    st.markdown("### Upload Training Data")
    uploaded_file = st.file_uploader("Upload sales transcript data (CSV or JSON)", type=["csv", "json"])
    
    if uploaded_file is not None:
        # Save and process the file
        with st.spinner("Processing uploaded file..."):
            result = upload_transcripts(uploaded_file)
            if "processed_path" in result:
                st.success(f"File uploaded and processed successfully!")
                dataset_path = result["processed_path"]
            else:
                st.error("Error processing file.")
                dataset_path = ""
    else:
        # Use existing dataset
        st.markdown("### Or use existing dataset")
        dataset_path = st.text_input("Dataset path", "./data/processed/sales_data.csv")
    
    # Training parameters
    st.markdown("### Training Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Epochs", min_value=1, max_value=50, value=10)
        batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)
    
    with col2:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
            format_func=lambda x: f"{x:.0e}",
            value=2e-5
        )
        eval_split = st.slider("Evaluation Split", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    
    max_seq_length = st.slider("Max Sequence Length", min_value=128, max_value=1024, value=512, step=64)
    
    # Start training
    if st.button("Start Training") and dataset_path:
        with st.spinner("Starting training process..."):
            params = {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_seq_length": max_seq_length,
                "eval_split": eval_split
            }
            result = train_model(dataset_path, params)
            if "message" in result:
                st.success(result["message"])
                # Poll metrics endpoint until available, then redirect
                poll_bar = st.progress(0)
                start_time = time.time()
                while True:
                    metrics = get_metrics()
                    if metrics and "accuracy" in metrics:
                        st.success("Training complete ✅. Redirecting to Evaluation page...")
                        st.session_state["nav"] = "Evaluation"
                        st.rerun()
                    # Update progress bar cyclically
                    poll_bar.progress(((time.time()-start_time)%15)/15)
                    time.sleep(5)
            else:
                st.error("Error starting training.")

# Evaluation page
elif page == "Evaluation":
    st.markdown('<p class="main-header">Model Evaluation</p>', unsafe_allow_html=True)
    
    # Refresh button
    if st.button("Refresh Metrics"):
        st.rerun()
    
    # Get metrics
    with st.spinner("Loading metrics..."):
        metrics = get_metrics()
    
    if metrics and "accuracy" in metrics:
        # Display metrics
        st.markdown("### Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col3.metric("Recall", f"{metrics['recall']:.2%}")
        col4.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        col5.metric("AUC-ROC", f"{metrics['auc_roc']:.2%}")
        
        # Model comparison
        st.markdown("### Model Comparison")
        
        with st.spinner("Loading comparison data..."):
            comparison = compare_models()
        
        if comparison:
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
            x = np.arange(len(metrics_to_plot))
            width = 0.35
            
            fine_tuned_values = [comparison["fine_tuned"][m] for m in metrics_to_plot]
            generic_values = [comparison["generic"][m] for m in metrics_to_plot]
            
            ax.bar(x - width/2, fine_tuned_values, width, label='Fine-tuned Model')
            ax.bar(x + width/2, generic_values, width, label='Generic Model')
            
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_title('Fine-tuned vs Generic Model Performance')
            ax.legend()
            
            st.pyplot(fig)
            
            # Improvement metrics
            st.markdown("### Performance Improvement")
            improvement_data = []
            for metric in metrics_to_plot:
                fine_tuned = comparison["fine_tuned"][metric]
                generic = comparison["generic"][metric]
                improvement = ((fine_tuned - generic) / generic) * 100 if generic > 0 else 0
                improvement_data.append({
                    "Metric": metric.replace("_", " ").title(),
                    "Improvement": improvement
                })
            
            improvement_df = pd.DataFrame(improvement_data)
            
            # Plot improvement
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Metric", y="Improvement", data=improvement_df, ax=ax)
            ax.set_title("Percentage Improvement over Generic Model")
            ax.set_ylabel("Improvement (%)")
            
            # Add value labels
            for i, p in enumerate(ax.patches):
                ax.annotate(f"{p.get_height():.1f}%", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'bottom', 
                            xytext = (0, 5), textcoords = 'offset points')
            
            st.pyplot(fig)
    else:
        st.warning("No metrics available. Please train the model first.")

# About page
elif page == "About":
    st.markdown('<p class="main-header">About</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Fine-Tuned Embeddings for Sales Conversion Prediction</h3>
    <p>This system uses domain-specific fine-tuned embeddings to improve sales conversion prediction accuracy from call transcripts.</p>
    
    <h4>Problem Statement</h4>
    <p>Sales teams struggle to accurately predict customer conversion likelihood from call transcripts, relying on subjective human judgment that leads to inconsistent predictions and missed opportunities. Generic embeddings fail to capture domain-specific sales nuances like buying signals, objection patterns, and conversation dynamics critical for accurate conversion assessment.</p>
    
    <h4>Solution Approach</h4>
    <ul>
        <li><strong>Domain-Specific Fine-Tuning:</strong> Fine-tune pre-trained embeddings on sales conversation data with conversion labels to capture sales-specific semantic patterns</li>
        <li><strong>Contrastive Learning:</strong> Train embeddings to distinguish between high-conversion and low-conversion conversation patterns</li>
        <li><strong>LangChain Framework:</strong> Orchestrate the fine-tuning pipeline, embedding generation, and similarity-based prediction workflow</li>
        <li><strong>Evaluation Pipeline:</strong> Compare fine-tuned vs. generic embeddings on conversion prediction tasks</li>
    </ul>
    </div>
    
    <div class="info-box">
    <h3>Technical Implementation</h3>
    <p>The system is built using:</p>
    <ul>
        <li><strong>FastAPI:</strong> Backend API for model serving and training</li>
        <li><strong>Streamlit:</strong> Interactive frontend for user interaction</li>
        <li><strong>Sentence Transformers:</strong> Base models for embeddings</li>
        <li><strong>LangChain:</strong> Framework for embedding generation and workflow orchestration</li>
        <li><strong>PyTorch:</strong> Deep learning framework for model fine-tuning</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)