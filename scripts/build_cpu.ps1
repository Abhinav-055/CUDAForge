param([string]$BuildDir = "build-cpu")
$ErrorActionPreference = "Stop"
cmake -S . -B $BuildDir -DENABLE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build $BuildDir --config Release
ctest --test-dir $BuildDir -C Release --output-on-failure
