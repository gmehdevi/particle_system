#!/bin/bash

# # Update and install basic Python and OpenGL dependencies
# sudo apt-get update
# sudo apt-get install -y build-essential python3-dev python3-pip python3-numpy python3-pyopengl libgl1-mesa-dev

# python3 -m pip install pybind11

# # Install PyOpenGL-accelerate for better performance
# pip install pyopengl-accelerate

# # Install additional dependencies for PyOpenCL
# sudo apt-get install -y git libboost-all-dev opencl-headers ocl-icd-opencl-dev ocl-icd-libopencl1
sudo pip install pybind11



rm -rf pyopencl
# Clone the PyOpenCL repository
git clone https://github.com/inducer/pyopencl.git

sudo git submodule update --init --recursive

# Navigate to the PyOpenCL directory
cd pyopencl

# Configure PyOpenCL with OpenGL interoperability
sudo ./configure.py --cl-enable-gl

# Build and install PyOpenCL
sudo ./setup.py build
sudo python3 setup.py install

# Test for PyOpenCL with OpenGL support
python3 -c "import pyopencl as cl; print('OpenGL support in PyOpenCL:', cl.have_gl())"

# Reminder for manual installation of an OpenCL SDK
echo "Remember to manually install the appropriate OpenCL SDK for your hardware."

# End of the script
