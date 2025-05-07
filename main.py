#!/usr/bin/env python
"""
Entry point for Yellow Sticky Trap Insect Detection
This script redirects to the main script in the src directory
"""
import sys
import os

if __name__ == "__main__":
    # Add the src directory to the Python path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    
    # Import and run the main function from src/main.py
    from src.main import main
    main()