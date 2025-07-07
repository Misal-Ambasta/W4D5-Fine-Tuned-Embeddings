from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.fine_tuning import FineTuner
from src.evaluation.metrics import ModelEvaluator
from src.data.data_processor import DataProcessor

app = FastAPI(
    title="Sales Conversion Prediction API",
    description="API for fine-tuned embeddings to predict sales conversion likelihood",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
embedding_model = EmbeddingModel()
fine_tuner = FineTuner()
model_evaluator = ModelEvaluator()
data_processor = DataProcessor()

# Pydantic models for request/response validation
class TranscriptRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class TrainingRequest(BaseModel):
    dataset_path: str
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    eval_split: float = 0.2

class PredictionResponse(BaseModel):
    conversion_probability: float
    embedding: List[float]
    similar_examples: List[Dict[str, Any]]

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    comparison: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Sales Conversion Prediction API"}

@app.post("/api/predict", response_model=PredictionResponse)
def predict_conversion(request: TranscriptRequest):
    try:
        prediction = embedding_model.get_prediction(request.text)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    try:
        # Start training in the background
        background_tasks.add_task(
            fine_tuner.train,
            dataset_path=request.dataset_path,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            max_seq_length=request.max_seq_length,
            eval_split=request.eval_split
        )
        
        return {"message": "Training started in the background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics", response_model=MetricsResponse)
def get_metrics():
    try:
        metrics = model_evaluator.get_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/compare")
def compare_models():
    try:
        comparison = model_evaluator.compare_models()
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/transcripts")
def upload_transcripts(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = os.path.join("data", "raw", file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        # Process the uploaded file
        processed_path = data_processor.process_transcripts(file_path)
        
        return {"message": "File uploaded and processed successfully", "processed_path": processed_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)