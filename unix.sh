#!/bin/bash

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

# Create a virtual environment
python3 -m venv particle_system_env

# Activate the virtual environment
source particle_system_env/bin/activate

# Install dependencies
pip install glfw PyOpenGL pyopencl PyGLM numpy

# Run the Python script
python your_script.py

# Deactivate the virtual environment
deactivate
