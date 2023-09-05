# Check if Python3 is installed
if (-not (Test-Path -Path "C:\Python39\python.exe" -PathType Leaf)) {
    Write-Host "Python3 is not installed. Please install Python3 and try again."
    exit 1
}

# Create a virtual environment
python -m venv particle_system_env

# Activate the virtual environment
.\particle_system_env\Scripts\Activate.ps1

# Install dependencies
pip install glfw PyOpenGL pyopencl PyGLM numpy

# Run the Python script
python simulation.py

# Deactivate the virtual environment
deactivate
