# AutoDL One-Click CUDA COLMAP + 3D Gaussian Splatting Pipeline

This repository provides a one-click workflow for running **CUDA COLMAP + 3D Gaussian Splatting** on AutoDL GPU servers.

The goal is simple:

1. On a fresh AutoDL server, run one setup script.
2. Upload a dataset zip file, such as `tower_scene.zip`.
3. Run one command.
4. Download the final output zip file, such as `tower_scene_output.zip`.

The pipeline automatically performs:

```text
Unzip dataset
→ organize images
→ CUDA COLMAP feature extraction
→ CUDA COLMAP feature matching
→ COLMAP mapper
→ COLMAP image undistortion
→ fix sparse folder structure for 3DGS
→ 3DGS training
→ zip final output
```

This workflow avoids several common problems:

- AutoDL system `colmap` may be compiled **without CUDA**.
- 3DGS official `convert.py` may accidentally call the CPU-only system COLMAP.
- `image_undistorter` may generate a `sparse` folder layout that 3DGS cannot directly read.
- `conda` may get stuck at `Solving environment`.
- Some AutoDL images do not include `nano`.

This repository solves these by:

- Compiling a dedicated CUDA COLMAP from source.
- Always using the compiled CUDA COLMAP path directly.
- Avoiding 3DGS `convert.py`.
- Using `mamba` instead of plain `conda` for environment creation.
- Installing `nano`, `zip`, `unzip`, `xvfb`, `screen`, and other required tools.
- Automatically fixing the `sparse/0` structure before training.

---

## 1. Server Requirements

This workflow is designed for AutoDL GPU instances.

Recommended server environment:

```text
Ubuntu
NVIDIA GPU
CUDA devel image
nvcc available
20 vCPU or more
Enough data disk space
```

Check whether the CUDA compiler is available:

```bash
nvcc --version
```

If you see:

```text
command not found
```

then your current image is probably a CUDA runtime image, not a CUDA devel image. You should choose an AutoDL image that includes CUDA devel tools.

A good choice is usually something like:

```text
PyTorch + CUDA 11.8 devel
or
PyTorch + CUDA 12.x devel
```

---

## 2. Standard Working Directory

All files should be placed under:

```text
/root/autodl-tmp/
```

After setup and one training run, the directory may look like this:

```text
/root/autodl-tmp/
├── colmap_cuda_src/
├── gaussian-splatting/
├── setup_3dgs_env.sh
├── run_3dgs_oneclick.sh
├── tower_scene.zip
├── tower_scene/
│   └── input/
│       ├── 001.jpg
│       ├── 002.jpg
│       └── ...
├── tower_scene_output/
└── tower_scene_output.zip
```

---

## 3. Dataset Zip Format

Assume your scene is named:

```text
tower_scene
```

Then upload:

```text
tower_scene.zip
```

The recommended zip structure is:

```text
tower_scene.zip
└── input/
    ├── 001.jpg
    ├── 002.jpg
    ├── 003.jpg
    └── ...
```

This structure is also accepted:

```text
tower_scene.zip
├── 001.jpg
├── 002.jpg
├── 003.jpg
└── ...
```

The script will automatically organize the images into:

```text
/root/autodl-tmp/tower_scene/input/
```

Recommended naming rules:

```text
Use English letters, numbers, and underscores.
Avoid spaces.
Avoid Chinese characters.
Avoid brackets.
```

Good examples:

```text
tower_scene.zip
substation_001.zip
powerline_test.zip
scene_20260630.zip
```

Bad examples:

```text
my scene.zip
塔场景.zip
scene(1).zip
```

---

## 4. First-Time Setup on a Fresh Server

Upload these two scripts to:

```text
/root/autodl-tmp/
```

```text
setup_3dgs_env.sh
run_3dgs_oneclick.sh
```

Then run:

```bash
cd /root/autodl-tmp
bash setup_3dgs_env.sh
```

The setup script will:

1. Check `nvcc`.
2. Install system dependencies.
3. Install `nano`, `vim`, `zip`, `unzip`, `xvfb`, and `screen`.
4. Configure Git for unstable network conditions.
5. Build CUDA COLMAP from source.
6. Download the official Gaussian Splatting repository.
7. Install `mamba`.
8. Create the `gaussian_splatting` conda environment using `mamba`.
9. Check whether PyTorch can access CUDA.

Successful setup should show something like:

```text
COLMAP 3.9.1 ... with CUDA
cuda available: True
device: NVIDIA GeForce RTX 4090
```

If COLMAP shows:

```text
without CUDA
```

then CUDA COLMAP was not built correctly.

---

## 5. Daily Usage

After the environment is ready, upload your dataset zip file to:

```text
/root/autodl-tmp/
```

For example:

```text
/root/autodl-tmp/tower_scene.zip
```

Then run:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential
```

After the script finishes, download:

```text
/root/autodl-tmp/tower_scene_output.zip
```

---

## 6. One-Click Command Format

The command format is:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh DATASET_ZIP MATCHER_TYPE RESOLUTION
```

Example:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential
```

Full example:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential 1
```

The three arguments mean:

```text
tower_scene.zip    dataset zip file
sequential         COLMAP matching strategy
1                  optional 3DGS resolution setting
```

The third argument is optional.

---

## 7. `exhaustive` vs `sequential`

### 7.1 `exhaustive`

Run:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip exhaustive
```

Meaning:

```text
COLMAP matches every image pair.
```

For `N` images, the number of image pairs is approximately:

```text
N × (N - 1) / 2
```

Examples:

```text
100 images  → about 4,950 pairs
1000 images → about 499,500 pairs
```

Advantages:

```text
More complete matching.
Better for unordered image sets.
Better when image sequence order is unreliable.
```

Disadvantages:

```text
Very slow for large datasets.
Not recommended for thousands of images.
```

Recommended use:

```text
Small datasets.
Unordered photos.
When you want maximum matching coverage.
```

---

### 7.2 `sequential`

Run:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential
```

Meaning:

```text
COLMAP mainly matches nearby images according to file order.
```

The script uses:

```text
SequentialMatching.overlap = 20
```

This is suitable for:

```text
Drone image sequences.
Video frames.
Continuous object capture.
Circular capture around a scene.
Large datasets with correct image order.
```

Advantages:

```text
Much faster than exhaustive matching.
More suitable for hundreds or thousands of ordered images.
```

Disadvantages:

```text
May fail if image order is wrong.
May miss matches if there are large jumps in the sequence.
```

For most future drone or continuous capture projects, use:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential
```

---

## 8. What Does the Final `1` Mean?

You may run:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential
```

or:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential 1
```

The final `1` is passed to 3DGS as:

```bash
-r 1
```

Meaning:

```text
Use original image resolution for 3DGS training.
```

Without this parameter, 3DGS may automatically downscale large images. For example, if images are wider than about 1.6K pixels, 3DGS may print:

```text
Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
If this is not desired, please explicitly specify '--resolution/-r as 1'
```

So:

```text
No final 1:
    Large images may be downscaled.
    Training is faster.
    GPU memory usage is lower.
    Usually recommended.

With final 1:
    Use original resolution.
    More detail may be preserved.
    Training is slower.
    GPU memory usage is higher.
```

Recommended default:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential
```

High-quality test:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential 1
```

---

## 9. Why Not Use Official `convert.py`?

The official 3DGS repository provides:

```bash
python convert.py -s scene_path
```

However, this script calls:

```bash
colmap
```

from the system PATH.

On AutoDL, the system COLMAP may be:

```text
COLMAP ... without CUDA
```

In that case, COLMAP feature extraction and matching will not use the GPU, even if the server has an RTX 4090.

This repository avoids that problem by always calling the compiled CUDA COLMAP directly:

```text
/root/autodl-tmp/colmap_cuda_src/build/src/colmap/exe/colmap
```

This ensures COLMAP shows:

```text
with CUDA
```

---

## 10. Why Fix the `sparse` Folder?

After `image_undistorter`, COLMAP may write undistorted sparse files to:

```text
scene/sparse/cameras.bin
scene/sparse/images.bin
scene/sparse/points3D.bin
```

But 3DGS expects:

```text
scene/sparse/0/cameras.bin
scene/sparse/0/images.bin
scene/sparse/0/points3D.bin
```

If the folder is not fixed, 3DGS may read the original distorted OPENCV camera model and fail with:

```text
Colmap camera model not handled:
only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!
```

The one-click script automatically moves:

```text
Original mapper result:
scene/sparse/0/
```

to:

```text
scene/distorted/sparse/0/
```

Then moves the undistorted files into:

```text
scene/sparse/0/
```

This makes 3DGS read the correct undistorted camera model.

---

## 11. Monitoring

Open another terminal and run:

```bash
watch -n 1 nvidia-smi
```

Expected behavior:

```text
feature_extractor:
    GPU should be used.

matcher:
    GPU should be used.

mapper:
    GPU usage may be low or zero.
    This is normal because mapper mainly uses CPU.

train.py:
    GPU should be heavily used.
```

To check CPU usage:

```bash
top
```

To check COLMAP processes:

```bash
ps aux | grep colmap
```

---

## 12. Recommended Use with `screen`

Long jobs should be run inside `screen` to avoid browser disconnection stopping the task.

Start a screen session:

```bash
screen -S gs
```

Run the pipeline inside screen:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential
```

Detach without stopping:

```text
Ctrl + A
then press D
```

Return to the session:

```bash
screen -r gs
```

---

## 13. Common Problems

### 13.1 `nano: command not found`

The setup script installs `nano`.

Manual installation:

```bash
apt update
apt install -y nano
```

---

### 13.2 `conda` stuck at `Solving environment`

This repository uses `mamba` instead of plain `conda`.

The setup script installs mamba using:

```bash
conda install -n base -c conda-forge mamba -y
```

Then creates the environment with:

```bash
mamba env create --file environment.yml
```

This is usually much faster than:

```bash
conda env create --file environment.yml
```

---

### 13.3 GitHub clone failure

Possible errors:

```text
RPC failed
GnuTLS recv error
fatal: early EOF
fatal: index-pack failed
```

This usually means the GitHub connection was interrupted.

The setup script configures Git with:

```bash
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
```

You can simply rerun:

```bash
bash /root/autodl-tmp/setup_3dgs_env.sh
```

If needed, remove the incomplete repository:

```bash
cd /root/autodl-tmp
rm -rf gaussian-splatting
bash setup_3dgs_env.sh
```

---

### 13.4 COLMAP still shows `without CUDA`

Check:

```bash
/root/autodl-tmp/colmap_cuda_src/build/src/colmap/exe/colmap -h | head
```

Correct result should include:

```text
with CUDA
```

Incorrect result:

```text
without CUDA
```

If incorrect, check:

```bash
nvcc --version
```

and make sure you are using a CUDA devel image.

---

### 13.5 3DGS camera model error

Error:

```text
Colmap camera model not handled:
only undistorted datasets supported!
```

This usually means 3DGS is reading the wrong `sparse/0` files.

The one-click script already fixes this automatically.

---

## 14. Quick Start Summary

Fresh server:

```bash
cd /root/autodl-tmp
bash setup_3dgs_env.sh
```

Upload:

```text
tower_scene.zip
```

Run:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential
```

Download:

```text
/root/autodl-tmp/tower_scene_output.zip
```

Small unordered dataset:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip exhaustive
```

Large ordered drone dataset:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential
```

Original-resolution 3DGS training:

```bash
bash /root/autodl-tmp/run_3dgs_oneclick.sh tower_scene.zip sequential 1
```

---

## 15. License

This repository only provides automation scripts. The external projects used by this workflow, such as COLMAP and Gaussian Splatting, have their own licenses. Please refer to their official repositories for details.
