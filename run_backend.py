#!/usr/bin/env python3
"""
Script to run the ACAS backend server from the project root.
This ensures proper module resolution.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(project_root, 'backend')

# Insert the project root at the beginning of sys.path
sys.path.insert(0, project_root)

# Now we can import and run the application
from backend.app import app

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[backend_path]
    )