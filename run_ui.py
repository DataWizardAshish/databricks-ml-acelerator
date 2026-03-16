"""Entry point: starts the Streamlit UI."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])
