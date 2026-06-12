"""Main entry point for running the skin cancer detection Flask app."""
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
