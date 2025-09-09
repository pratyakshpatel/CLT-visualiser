#!/usr/bin/env python3
"""
Simple runner script for the CLT Visualizer
This script can be run in a Jupyter notebook or as a standalone script
"""

# Import the CLT Visualizer
from clt_visualizer import CLTVisualizer

def main():
    print("Starting CLT Visualizer...")
    print("=" * 50)
    print("Central Limit Theorem Interactive Visualizer")
    print("=" * 50)
    print("\nThis visualizer demonstrates the Central Limit Theorem with:")
    print("• 8 different probability distributions")
    print("• Interactive parameter controls")
    print("• Real-time visualization updates")
    print("• Comparison of theoretical vs observed results")
    print("\nInstructions:")
    print("1. Use the dropdown to select a distribution")
    print("2. Adjust parameters with sliders")
    print("3. Change sample size and number of samples")
    print("4. Observe how sample means approach normality!")
    print("\n" + "=" * 50)
    
    # Create and display the visualizer
    visualizer = CLTVisualizer()
    visualizer.display()

if __name__ == "__main__":
    main()