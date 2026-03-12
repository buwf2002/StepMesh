bash tools/install_deps.sh # only once
USE_CUDA=0 USE_HIP=1 make af
USE_CUDA=0 USE_HIP=1 pip3 install -v -e . --no-build-isolation
