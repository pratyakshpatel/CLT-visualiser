#!/bin/bash

# setup script for the CLT visualizer
echo "setting up central limit theorem visualizer..."

# check if python is available
if ! command -v python3 &> /dev/null
then
    echo "python3 is required but not installed"
    echo "please install python 3.7 or higher"
    exit 1
fi

echo "python3 found: $(python3 --version)"

# run the basic demo first
echo ""
echo "running basic demo (no dependencies needed)..."
python3 main.py

echo ""
echo "setup complete!"
echo ""
echo "to run again: python3 main.py"
echo "for interactive version: pip install -r requirements.txt && python3 clt_visualizer.py"
echo "for jupyter version: pip install jupyter numpy matplotlib scipy ipywidgets && jupyter lab CLT_Visualizer.ipynb"