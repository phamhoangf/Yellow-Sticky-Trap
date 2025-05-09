"""
Launcher script for the Yellow Sticky Trap detection Streamlit app
"""
import os
import subprocess
import sys

def launch_app():
    """Launch the Streamlit app"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the Streamlit app
    app_path = os.path.join(script_dir, "app.py")
      # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.maxUploadSize=200", 
            "--browser.serverAddress=localhost",
            "--server.port=8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Streamlit app: {e}")
    except FileNotFoundError:
        print("Streamlit is not installed. Please install it using:")
        print("pip install streamlit pillow")

if __name__ == "__main__":
    launch_app()
