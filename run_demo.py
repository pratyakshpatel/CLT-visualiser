#!/usr/bin/env python3
"""
Quick runner for CLT demonstration
"""

import sys
import subprocess

def check_and_install_requirements():
    """Check if required packages are installed, install if needed"""
    required_packages = ['numpy', 'matplotlib', 'scipy']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} not found. Installing...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}. Please install manually: pip install {package}")
                return False
    return True

def run_clt_demo():
    """Run the CLT demonstration"""
    print("Starting CLT Visualizer...")
    print("="*50)
    
    try:
        # Try to run the simple demo
        import simple_clt_demo
        print("CLT Demo completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required packages are installed.")
        print("Run: pip install numpy matplotlib scipy")
        
    except Exception as e:
        print(f"Error running demo: {e}")

if __name__ == "__main__":
    print("CLT Visualizer Setup")
    print("="*30)
    
    if check_and_install_requirements():
        print("\nAll requirements satisfied!")
        run_clt_demo()
    else:
        print("\nSome requirements missing. Please install manually and try again.")