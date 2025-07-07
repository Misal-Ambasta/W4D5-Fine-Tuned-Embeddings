# Fine-Tuned Embeddings for Sales Conversion Prediction

## Overview
This project implements a system for predicting sales conversion probability from call transcripts using fine-tuned embeddings. It addresses the limitations of generic embeddings by fine-tuning pre-trained models specifically for sales conversations using contrastive learning.

## Features
- Fine-tuning of pre-trained embedding models using contrastive learning
- Sales conversion prediction from call transcripts
- Performance comparison between generic and fine-tuned embeddings
- Interactive web interface for model training and evaluation
- RESTful API for integration with other systems

## Project Structure
```
├── api/                  # FastAPI backend
├── app/                  # Streamlit frontend
├── data/                 # Data storage
│   ├── raw/              # Raw input data
│   └── processed/        # Processed data
├── models/               # Model storage
│   ├── base/             # Pre-trained models
│   └── fine_tuned/       # Fine-tuned models
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── data/             # Data processing modules
│   ├── embeddings/       # Embedding generation and fine-tuning
│   ├── evaluation/       # Model evaluation
│   └── utils/            # Utility functions
├── tests/                # Test cases
├── .env.example          # Example environment variables
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Installation

1. Clone the repository

2. Create a virtual environment
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment
   - Windows:
   ```bash
   .venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables
   ```bash
   cp .env.example .env
   ```

6. Run the setup script to create directories and generate sample data
   ```bash
   python setup.py
   ```

## Usage

### FastAPI Backend

Start the API server:
```bash
python run_api.py
```

The API will be available at http://localhost:8000

### Streamlit Frontend

Start the Streamlit app:
```bash
python run_app.py
```

The app will be available at http://localhost:8501

### Run Both Services

Start both the API and Streamlit app:
```bash
python run_all.py
```

## API Endpoints

- `POST /predict`: Predict conversion probability from a transcript
- `POST /train`: Train a fine-tuned embedding model
- `GET /metrics`: Get model performance metrics
- `GET /compare`: Compare base and fine-tuned model performance
- `POST /upload`: Upload a new transcript with conversion outcome

## Evaluation

Run the evaluation script to compare the performance of base and fine-tuned models:
```bash
python run_evaluation.py
```

## Testing

Run the test suite:
```bash
python run_tests.py
```

## Generate Sample Data

Generate additional sample data for testing:
```bash
python generate_sample_data.py [num_samples]
```

## License
MIT

## About the Solution

This solution uses domain-specific fine-tuning of pre-trained embeddings on sales conversation data with conversion labels to capture sales-specific semantic patterns. It employs contrastive learning to train embeddings to distinguish between high-conversion and low-conversion conversation patterns.

The implementation orchestrates the entire pipeline using functional programming principles, from data processing to model training and evaluation. The system provides both a programmatic API and an intuitive web interface for sales teams to leverage improved conversion predictions.

## Features

- Fine-tuning pipeline for domain-specific embeddings
- Contrastive learning implementation
- LangChain integration for embedding generation and workflow orchestration
- Evaluation metrics comparing fine-tuned vs. generic embeddings
- FastAPI backend for model serving
- Streamlit frontend for interactive usage

## Installation

1. Clone the repository
2. Create virtual enviornment
   ```
   python -m venv .venv
   ``` 
3. Activate virtual environment
   ```
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Copy `.env.example` to `.env` and configure as needed

## Usage

### 1. Run Backend & Front-end Together (recommended)

`run_all.py` spins up **both** the FastAPI backend and the Streamlit front-end for you.

```bash
# Windows activation script
python run_all.py --win

# Linux / macOS activation script
python run_all.py --linux
```

Optional flags:

* `--api-port <PORT>` – backend port (default `8000`)
* `--app-port <PORT>` – front-end port (default `8501`)
* `--no-venv` – skip virtual-environment activation if you already activated `.venv`

Example:
```bash
python run_all.py --win --api-port 8001 --app-port 8502
```

`run_all.py` sets `API_PORT` and `STREAMLIT_PORT` environment variables internally so the two services automatically talk to each other.

---

### 2. Run Services Manually (advanced)

If you prefer to start the services yourself:

**Backend**
```bash
uvicorn api.main:app --reload --port 8000
```

**Front-end**
```bash
streamlit run app/main.py
```

Make sure the `API_PORT` env-var (or `.env` file) matches the port you start FastAPI on.

## API Endpoints

- `POST /api/predict`: Get conversion prediction for a sales transcript
- `POST /api/train`: Trigger fine-tuning of embeddings
- `GET /api/metrics`: Get model performance metrics
- `GET /api/compare`: Compare fine-tuned vs generic embeddings

## License

MIT
