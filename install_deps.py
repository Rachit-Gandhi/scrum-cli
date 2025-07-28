#!/usr/bin/env python3
"""Install dependencies for SCRUMS-CLI"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show progress"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ {description} failed with unexpected error: {e}")
        return False

def main():
    """Install dependencies"""
    print("🚀 Installing SCRUMS-CLI Dependencies")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    print(f"📁 Working directory: {project_dir}")
    
    # Install core dependencies first
    core_deps = [
        "click>=8.0.0",
        "fastapi>=0.116.1", 
        "google-generativeai>=0.7.0",
        "requests>=2.28.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "uvicorn>=0.35.0",
    ]
    
    print(f"\n📦 Installing core dependencies...")
    for dep in core_deps:
        if not run_command(f"pip install '{dep}'", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️ Failed to install {dep}, but continuing...")
    
    # Try to install the project in development mode
    print(f"\n🔧 Installing project in development mode...")
    if run_command("pip install -e .", "Installing project"):
        print("✅ Project installation completed")
    else:
        print("⚠️ Project installation failed, trying alternative...")
        if run_command("python setup.py develop", "Alternative installation"):
            print("✅ Alternative installation completed")
        else:
            print("❌ Both installation methods failed")
    
    # Test if CLI works
    print(f"\n🧪 Testing CLI installation...")
    if run_command("scrum-cli --help", "Testing CLI"):
        print("🎉 SCRUMS-CLI is ready to use!")
        print("\n📋 Try these commands:")
        print("   scrum-cli --help")
        print("   scrum-cli status") 
        print("   scrum-cli live-meeting --demo-mode")
        print("   scrum-cli host-proxy --help")
    else:
        print("⚠️ CLI test failed - manual intervention may be needed")
        print("\n🔧 Try these manual steps:")
        print("   1. pip install -e .")
        print("   2. Set PYTHONPATH if needed")
        print("   3. Check your Python environment")
    
    print(f"\n✨ Installation script completed!")

if __name__ == "__main__":
    main()