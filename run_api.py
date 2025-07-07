import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables
host = os.getenv("API_HOST", "0.0.0.0")
port = int(os.getenv("API_PORT", "8000"))

def run_api():
    """Run the FastAPI backend"""
    print(f"Starting API server at http://{host}:{port}")
    uvicorn.run("api.main:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    run_api()