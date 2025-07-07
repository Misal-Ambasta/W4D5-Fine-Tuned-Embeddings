import streamlit.web.cli as stcli
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

def run_streamlit():
    """Run the Streamlit frontend"""
    # Get port from environment variable or use default
    port = os.getenv("STREAMLIT_SERVER_PORT", "8501")
    print(f"Starting Streamlit app on port {port}")
    sys.argv = ["streamlit", "run", "app/main.py", f"--server.port={port}"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    run_streamlit()