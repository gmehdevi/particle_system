# Clone the PyOpenCL repository
git clone https://github.com/inducer/pyopencl.git
cd pyopencl
git submodule update --init --recursive

python3 configure.py --cl-enable-gl
python3 setup.py build_ext --include-dirs="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include" --library-dirs="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64" --define="HAVE_GL=1"
python3 setup.py install
# python3 -m pip install .
cd ..

pip install -r requirements.txt 