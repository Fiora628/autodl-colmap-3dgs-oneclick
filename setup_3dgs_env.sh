#!/bin/bash
set -e

WORKDIR="/root/autodl-tmp"
COLMAP_SRC="$WORKDIR/colmap_cuda_src"
COLMAP_BIN="$COLMAP_SRC/build/src/colmap/exe/colmap"
GS_DIR="$WORKDIR/gaussian-splatting"

# RTX 4090 uses CUDA architecture 89.
# For another GPU, override this variable before running:
# export COLMAP_CUDA_ARCH=86
COLMAP_CUDA_ARCH="${COLMAP_CUDA_ARCH:-89}"

echo "============================================================"
echo "Setup CUDA COLMAP + 3DGS environment"
echo "Workdir: $WORKDIR"
echo "COLMAP CUDA arch: $COLMAP_CUDA_ARCH"
echo "============================================================"

retry() {
    local n=1
    local max=3
    local delay=5

    while true; do
        echo "Running command, attempt $n/$max:"
        echo "$@"
        "$@" && break || {
            if [ "$n" -lt "$max" ]; then
                n=$((n+1))
                echo "Command failed. Retrying in $delay seconds..."
                sleep "$delay"
            else
                echo "Command failed after $max attempts."
                return 1
            fi
        }
    done
}

echo ""
echo "[1/8] Checking CUDA compiler..."

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found."
    echo "This instance is probably a CUDA runtime image, not a CUDA devel image."
    echo "Please choose an AutoDL image with CUDA devel, for example CUDA 11.8/12.x devel + PyTorch."
    exit 1
fi

nvcc --version

echo ""
echo "[2/8] Installing system dependencies..."

apt update
apt install -y \
    git cmake ninja-build build-essential \
    nano vim unzip zip xvfb screen \
    ca-certificates curl wget \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libboost-filesystem-dev libboost-iostreams-dev libboost-regex-dev \
    libboost-test-dev libboost-thread-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev libsuitesparse-dev \
    libcurl4-openssl-dev libssl-dev

echo ""
echo "[3/8] Configuring Git for unstable networks..."

git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999

echo ""
echo "[4/8] Building CUDA COLMAP..."

cd "$WORKDIR"

if [ ! -d "$COLMAP_SRC/.git" ]; then
    rm -rf "$COLMAP_SRC"
    retry git clone --depth 1 --branch 3.9.1 https://github.com/colmap/colmap.git "$COLMAP_SRC"
fi

cd "$COLMAP_SRC"

rm -rf build
mkdir -p build
cd build

cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$COLMAP_CUDA_ARCH" \
    -DGUI_ENABLED=OFF

ninja -j "$(nproc)" || ninja -j 8

if [ ! -x "$COLMAP_BIN" ]; then
    echo "ERROR: CUDA COLMAP build failed. Executable not found:"
    echo "$COLMAP_BIN"
    exit 1
fi

echo ""
echo "[5/8] Checking CUDA COLMAP..."

"$COLMAP_BIN" -h | head -5

if ! "$COLMAP_BIN" -h | head -5 | grep -q "with CUDA"; then
    echo "ERROR: COLMAP was built, but it does not show 'with CUDA'."
    exit 1
fi

echo ""
echo "[6/8] Installing Gaussian Splatting repository..."

cd "$WORKDIR"

clone_gs() {
    rm -rf "$GS_DIR"
    git clone --depth 1 --recursive --shallow-submodules https://github.com/graphdeco-inria/gaussian-splatting.git "$GS_DIR"
}

if [ ! -d "$GS_DIR/.git" ]; then
    retry clone_gs
fi

cd "$GS_DIR"

retry git submodule update --init --recursive --depth 1

echo ""
echo "[7/8] Installing mamba and creating conda environment..."

if [ ! -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "ERROR: Cannot find conda init script:"
    echo "/root/miniconda3/etc/profile.d/conda.sh"
    echo "Please check your AutoDL image."
    exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh

if ! command -v mamba >/dev/null 2>&1; then
    echo "mamba not found. Installing mamba..."
    conda install -n base -c conda-forge mamba -y
fi

hash -r

if conda env list | awk '{print $1}' | grep -qx "gaussian_splatting"; then
    echo "Conda env gaussian_splatting already exists. Skipping creation."
else
    mamba env create --file environment.yml
fi

conda activate gaussian_splatting

echo ""
echo "[8/8] Checking PyTorch CUDA..."

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access CUDA.")
PY

echo "============================================================"
echo "Setup complete."
echo ""
echo "CUDA COLMAP:"
echo "$COLMAP_BIN"
echo ""
echo "Gaussian Splatting:"
echo "$GS_DIR"
echo ""
echo "Next step:"
echo "Upload your scene zip to /root/autodl-tmp/"
echo "Then run:"
echo "bash /root/autodl-tmp/run_3dgs_oneclick.sh your_scene.zip sequential"
echo "============================================================"
