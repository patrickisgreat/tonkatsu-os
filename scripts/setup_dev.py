#!/usr/bin/env python3
"""
Development environment setup script for Tonkatsu-OS.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=cwd, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Command: {cmd}")
        print(f"Error: {e.stderr}")
        return False

def main():
    project_root = Path(__file__).parent.parent
    frontend_path = project_root / "frontend"
    
    print("🚀 Setting up Tonkatsu-OS Development Environment")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python 3.8+ required. Current version: {python_version.major}.{python_version.minor}")
        return 1
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if Poetry is installed
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        print("✅ Poetry is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Poetry not found. Please install Poetry first:")
        print("   curl -sSL https://install.python-poetry.org | python3 -")
        return 1
    
    # Install Python dependencies
    os.chdir(project_root)
    if not run_command("poetry install", "Installing Python dependencies"):
        return 1
    
    # Setup frontend
    if frontend_path.exists():
        print("\n📦 Setting up frontend...")
        
        # Check if Node.js is installed
        try:
            subprocess.run(["node", "--version"], check=True, capture_output=True)
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
            print("✅ Node.js and npm are installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Node.js/npm not found. Please install Node.js first:")
            print("   https://nodejs.org/")
            return 1
        
        # Install frontend dependencies
        if not run_command("npm install", "Installing frontend dependencies", cwd=frontend_path):
            print("⚠️  Frontend setup failed, but backend will still work")
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = [
        "data/raw",
        "data/processed", 
        "data/models",
        "logs",
        "exports"
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Create startup scripts permissions
    print("\n🔐 Setting script permissions...")
    scripts = [
        "scripts/start_backend.py",
        "scripts/start_frontend.py"
    ]
    
    for script in scripts:
        script_path = project_root / script
        if script_path.exists():
            os.chmod(script_path, 0o755)
            print(f"✅ Made executable: {script}")
    
    print("\n" + "=" * 60)
    print("🎉 Development environment setup complete!")
    print("")
    print("To start the application:")
    print("1. Backend:  poetry run python scripts/start_backend.py")
    print("2. Frontend: python scripts/start_frontend.py")
    print("")
    print("Or use the convenience commands:")
    print("• make dev-backend")
    print("• make dev-frontend")
    print("• make dev (starts both)")
    print("")
    print("Access points:")
    print("• Frontend: http://localhost:3000")
    print("• Backend API: http://localhost:8000")
    print("• API Docs: http://localhost:8000/docs")

if __name__ == "__main__":
    sys.exit(main())