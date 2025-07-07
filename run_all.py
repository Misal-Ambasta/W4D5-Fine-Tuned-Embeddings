import subprocess
import sys
import os
import time
import platform
import socket
import argparse
from pathlib import Path

def activate_venv():
    """Activate the virtual environment based on the operating system"""
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Virtual environment not found at .venv. Please create it first.")
        return False
        
    # Determine the path to the activation script based on OS
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
    else:  # Unix-like systems (Linux, macOS)
        activate_script = venv_path / "bin" / "activate"
        
    if not activate_script.exists():
        print(f"Activation script not found at {activate_script}")
        return False
        
    print(f"Activating virtual environment from {activate_script}")
    return str(activate_script)

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_all():
    """Run both the FastAPI backend and Streamlit frontend"""
    print("Starting Fine-Tuned Embeddings for Sales Conversion Prediction...")
    
    # Get port configuration from environment variables
    api_port = int(os.environ.get('API_PORT', 8000))
    app_port = int(os.environ.get('STREAMLIT_PORT', 8501))
    
    # Check if ports are available
    if is_port_in_use(api_port):
        print(f"Warning: Port {api_port} is already in use. The API server might fail to start.")
        print(f"Try using a different port with --api-port option.")
    
    if is_port_in_use(app_port):
        print(f"Warning: Port {app_port} is already in use. The Streamlit app might fail to start.")
        print(f"Try using a different port with --app-port option.")
    
    # Check if already inside .venv
    venv_path = os.path.abspath('.venv')
    # Safe check: is sys.prefix or VIRTUAL_ENV pointing to .venv?
    def is_venv_active():
        venv_env = os.environ.get('VIRTUAL_ENV')
        if venv_env and os.path.abspath(venv_env) == venv_path:
            return True
        # On Windows, sys.prefix may be inside .venv, or equal to it
        try:
            return os.path.abspath(sys.prefix) == venv_path or os.path.abspath(sys.prefix).startswith(venv_path + os.sep)
        except Exception:
            return False
    already_in_venv = is_venv_active()

    if os.environ.get('SKIP_VENV') == 'true' or already_in_venv:
        if already_in_venv:
            print("Virtual environment is already activated. Skipping activation.")
        else:
            print("Skipping virtual environment activation as requested.")
        venv_prefix = []
    else:
        venv_activate = activate_venv()
        if not venv_activate:
            print("Failed to activate virtual environment. Continuing without it...")
            venv_prefix = []
        else:
            # Use CLI argument to determine activation command
            if args.win:
                venv_prefix = ["cmd", "/c", f"call {venv_activate} &&"]
            elif args.linux:
                venv_prefix = ["bash", "-c", f"source {venv_activate} &&"]
            else:
                venv_prefix = []  # Should not happen due to mutually exclusive group
    
    # Start the FastAPI backend
    print(f"\nStarting FastAPI backend on port {api_port}...")
    if venv_prefix:
        # When using virtual environment
        if platform.system() == "Windows":
            api_cmd = venv_prefix + [sys.executable, "run_api.py"]
        else:
            api_cmd = venv_prefix + [f"python run_api.py"]
        api_process = subprocess.Popen(
            api_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=(platform.system() == "Windows"),
            env=dict(os.environ, API_PORT=str(api_port))
        )
    else:
        # Fallback to direct execution without virtual environment
        api_process = subprocess.Popen(
            [sys.executable, "run_api.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ, API_PORT=str(api_port))
        )
    
    # Wait a bit for the API to start
    print("Waiting for API to start...")
    time.sleep(3)
    
    # Check if the API process is still running
    if api_process.poll() is not None:
        print("Error starting API server:")
        print(api_process.stderr.read())
        return 1
    
    print(f"API server running at http://localhost:{api_port}")
    
    # Start the Streamlit frontend
    print(f"\nStarting Streamlit frontend on port {app_port}...")
    if venv_prefix:
        # When using virtual environment
        if platform.system() == "Windows":
            app_cmd = venv_prefix + [sys.executable, "run_app.py"]
        else:
            app_cmd = venv_prefix + [f"python run_app.py"]
        app_process = subprocess.Popen(
            app_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=(platform.system() == "Windows"),
            env=dict(os.environ, STREAMLIT_SERVER_PORT=str(app_port))
        )
    else:
        # Fallback to direct execution without virtual environment
        app_process = subprocess.Popen(
            [sys.executable, "run_app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ, STREAMLIT_SERVER_PORT=str(app_port))
        )
    
    # Wait a bit for the app to start
    print("Waiting for Streamlit app to start...")
    time.sleep(5)
    
    # Check if the app process is still running
    if app_process.poll() is not None:
        print("Error starting Streamlit app:")
        print(app_process.stderr.read())
        api_process.terminate()
        return 1
    
    print(f"Streamlit app running at http://localhost:{app_port}")
    
    print("\nBoth services are now running!")
    print("Press Ctrl+C to stop all services")
    
    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        api_process.terminate()
        app_process.terminate()
        print("Services stopped")
    
    return 0

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the Fine-Tuned Embeddings application")
    parser.add_argument("--api-port", type=int, default=int(os.environ.get('API_PORT', 8000)),
                        help="Port for the FastAPI backend (default: 8000)")
    parser.add_argument("--app-port", type=int, default=int(os.environ.get('STREAMLIT_PORT', 8501)),
                        help="Port for the Streamlit frontend (default: 8501)")
    parser.add_argument("--no-venv", action="store_true",
                        help="Skip virtual environment activation")
    os_group = parser.add_mutually_exclusive_group(required=True)
    os_group.add_argument("--win", action="store_true", help="Run using Windows activation command")
    os_group.add_argument("--linux", action="store_true", help="Run using Linux/Unix activation command")
    return parser.parse_args()

if __name__ == "__main__":
    # Set environment variables from command line arguments
    args = parse_arguments()
    os.environ['API_PORT'] = str(args.api_port)
    os.environ['STREAMLIT_PORT'] = str(args.app_port)
    
    # Skip virtual environment if requested
    if args.no_venv:
        os.environ['SKIP_VENV'] = 'true'

    sys.exit(run_all())