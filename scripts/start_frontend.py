#!/usr/bin/env python3
"""
Startup script for Tonkatsu-OS React frontend.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    frontend_path = project_root / "frontend"
    
    print("🌐 Starting Tonkatsu-OS Frontend...")
    print("=" * 50)
    print("Frontend URL: http://localhost:3000")
    print("Make sure backend is running on http://localhost:8000")
    print("=" * 50)
    
    # Check if we're in the frontend directory
    if not frontend_path.exists():
        print(f"❌ Frontend directory not found: {frontend_path}")
        return 1
    
    # Change to frontend directory
    os.chdir(frontend_path)
    
    # Check if node_modules exists
    if not (frontend_path / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        try:
            subprocess.run(["npm", "install"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Make sure Node.js and npm are installed.")
            return 1
        except FileNotFoundError:
            print("❌ npm not found. Please install Node.js and npm first.")
            return 1
    
    # Start the development server
    try:
        subprocess.run(["npm", "run", "dev"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to start frontend server.")
        return 1
    except FileNotFoundError:
        print("❌ npm not found. Please install Node.js and npm first.")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Frontend server stopped.")
        return 0

if __name__ == "__main__":
    sys.exit(main())