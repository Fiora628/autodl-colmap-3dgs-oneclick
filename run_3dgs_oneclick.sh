#!/bin/bash
set -e

# ============================================================
# One-click COLMAP CUDA + 3DGS pipeline
#
# Usage:
#   bash run_3dgs_oneclick.sh scene_name.zip [exhaustive|sequential] [resolution]
#
# Examples:
#   bash run_3dgs_oneclick.sh tower_scene.zip sequential
#   bash run_3dgs_oneclick.sh tower_scene.zip exhaustive
#   bash run_3dgs_oneclick.sh tower_scene.zip sequential 1
#
# Required:
#   zip file placed in /root/autodl-tmp/
#   CUDA COLMAP compiled at /root/autodl-tmp/colmap_cuda_src/build/src/colmap/exe/colmap
#   3DGS repo at /root/autodl-tmp/gaussian-splatting
# ============================================================

ZIP_NAME=$1
MATCHER_TYPE=$2
RESOLUTION=$3

WORKDIR="/root/autodl-tmp"
COLMAP_CUDA="$WORKDIR/colmap_cuda_src/build/src/colmap/exe/colmap"
GS_DIR="$WORKDIR/gaussian-splatting"
NUM_THREADS=$(nproc)

if [ -z "$ZIP_NAME" ]; then
    echo "ERROR: Please provide a zip file name."
    echo "Usage: bash run_3dgs_oneclick.sh scene_name.zip [exhaustive|sequential] [resolution]"
    exit 1
fi

if [ -z "$MATCHER_TYPE" ]; then
    MATCHER_TYPE="sequential"
fi

if [ -z "$RESOLUTION" ]; then
    RESOLUTION=""
fi

ZIP_PATH="$WORKDIR/$ZIP_NAME"

if [ ! -f "$ZIP_PATH" ]; then
    echo "ERROR: Zip file not found:"
    echo "$ZIP_PATH"
    exit 1
fi

if [ ! -x "$COLMAP_CUDA" ]; then
    echo "ERROR: CUDA COLMAP not found or not executable:"
    echo "$COLMAP_CUDA"
    echo "Please run setup_3dgs_env.sh first."
    exit 1
fi

if [ ! -d "$GS_DIR" ]; then
    echo "ERROR: gaussian-splatting repo not found:"
    echo "$GS_DIR"
    echo "Please run setup_3dgs_env.sh first."
    exit 1
fi

SCENE_BASE=$(basename "$ZIP_NAME")
SCENE_NAME="${SCENE_BASE%.*}"
SCENE_PATH="$WORKDIR/$SCENE_NAME"
OUTPUT_PATH="$WORKDIR/${SCENE_NAME}_output"
OUTPUT_ZIP="$WORKDIR/${SCENE_NAME}_output.zip"

echo "============================================================"
echo "One-click 3DGS pipeline"
echo "Zip file:      $ZIP_PATH"
echo "Scene name:    $SCENE_NAME"
echo "Scene path:    $SCENE_PATH"
echo "Output path:   $OUTPUT_PATH"
echo "Output zip:    $OUTPUT_ZIP"
echo "Matcher:       $MATCHER_TYPE"
echo "Resolution:    ${RESOLUTION:-default}"
echo "Threads:       $NUM_THREADS"
echo "COLMAP CUDA:   $COLMAP_CUDA"
echo "============================================================"

echo ""
echo "[0/9] Checking CUDA COLMAP version..."

"$COLMAP_CUDA" -h | head -5

if ! "$COLMAP_CUDA" -h | head -5 | grep -q "with CUDA"; then
    echo "ERROR: This COLMAP is not built with CUDA."
    exit 1
fi

echo ""
echo "[1/9] Cleaning old scene/output if they exist..."

rm -rf "$SCENE_PATH"
rm -rf "$OUTPUT_PATH"
rm -f "$OUTPUT_ZIP"

mkdir -p "$SCENE_PATH"

echo ""
echo "[2/9] Unzipping dataset..."

unzip -q "$ZIP_PATH" -d "$SCENE_PATH"

echo ""
echo "[3/9] Normalizing image folder structure..."

TOP_DIR_COUNT=$(find "$SCENE_PATH" -mindepth 1 -maxdepth 1 -type d | wc -l)
TOP_FILE_COUNT=$(find "$SCENE_PATH" -mindepth 1 -maxdepth 1 -type f | wc -l)

# Case A:
# zip contains one top-level folder, for example:
# tower_scene/input/*.jpg
if [ "$TOP_DIR_COUNT" -eq 1 ] && [ "$TOP_FILE_COUNT" -eq 0 ]; then
    ONLY_DIR=$(find "$SCENE_PATH" -mindepth 1 -maxdepth 1 -type d | head -1)

    if [ "$(basename "$ONLY_DIR")" != "input" ]; then
        echo "Detected single top-level folder: $ONLY_DIR"
        TMP_DIR="${SCENE_PATH}_tmp_unpack"
        rm -rf "$TMP_DIR"
        mv "$ONLY_DIR" "$TMP_DIR"
        find "$TMP_DIR" -mindepth 1 -maxdepth 1 -exec mv {} "$SCENE_PATH/" \;
        rm -rf "$TMP_DIR"
    fi
fi

# Case B:
# if images/ exists but input/ does not, rename images to input.
if [ ! -d "$SCENE_PATH/input" ] && [ -d "$SCENE_PATH/images" ]; then
    echo "Renaming images/ to input/"
    mv "$SCENE_PATH/images" "$SCENE_PATH/input"
fi

# Case C:
# if input/ does not exist, create it and move top-level image files into it.
if [ ! -d "$SCENE_PATH/input" ]; then
    echo "Creating input/ and moving image files into it..."
    mkdir -p "$SCENE_PATH/input"
    find "$SCENE_PATH" -maxdepth 1 -type f \( \
        -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \
    \) -exec mv {} "$SCENE_PATH/input/" \;
fi

IMAGE_COUNT=$(find "$SCENE_PATH/input" -type f \( \
    -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \
\) | wc -l)

echo "Image count: $IMAGE_COUNT"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "ERROR: No images found in:"
    echo "$SCENE_PATH/input"
    echo "Please check your zip file structure."
    exit 1
fi

echo ""
echo "[4/9] Cleaning old COLMAP files..."

rm -f "$SCENE_PATH/database.db"
rm -rf "$SCENE_PATH/sparse"
rm -rf "$SCENE_PATH/distorted"
rm -rf "$SCENE_PATH/images"

echo ""
echo "[5/9] Running COLMAP CUDA feature extraction..."

xvfb-run -a -s "-screen 0 1920x1080x24" "$COLMAP_CUDA" feature_extractor \
    --database_path "$SCENE_PATH/database.db" \
    --image_path "$SCENE_PATH/input" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.gpu_index 0

echo ""
echo "[6/9] Running COLMAP CUDA matching..."

if [ "$MATCHER_TYPE" = "exhaustive" ]; then
    xvfb-run -a -s "-screen 0 1920x1080x24" "$COLMAP_CUDA" exhaustive_matcher \
        --database_path "$SCENE_PATH/database.db" \
        --SiftMatching.use_gpu 1 \
        --SiftMatching.gpu_index 0
elif [ "$MATCHER_TYPE" = "sequential" ]; then
    xvfb-run -a -s "-screen 0 1920x1080x24" "$COLMAP_CUDA" sequential_matcher \
        --database_path "$SCENE_PATH/database.db" \
        --SiftMatching.use_gpu 1 \
        --SiftMatching.gpu_index 0 \
        --SequentialMatching.overlap 20
else
    echo "ERROR: matcher type must be exhaustive or sequential"
    exit 1
fi

echo ""
echo "[7/9] Running COLMAP mapper..."

mkdir -p "$SCENE_PATH/sparse"

xvfb-run -a -s "-screen 0 1920x1080x24" "$COLMAP_CUDA" mapper \
    --database_path "$SCENE_PATH/database.db" \
    --image_path "$SCENE_PATH/input" \
    --output_path "$SCENE_PATH/sparse" \
    --Mapper.num_threads "$NUM_THREADS"

if [ ! -d "$SCENE_PATH/sparse/0" ]; then
    echo "ERROR: mapper did not generate sparse/0"
    echo "Possible reasons: insufficient image overlap, wrong image order, too few matches."
    exit 1
fi

echo ""
echo "[8/9] Running COLMAP image undistorter..."

xvfb-run -a -s "-screen 0 1920x1080x24" "$COLMAP_CUDA" image_undistorter \
    --image_path "$SCENE_PATH/input" \
    --input_path "$SCENE_PATH/sparse/0" \
    --output_path "$SCENE_PATH" \
    --output_type COLMAP

echo ""
echo "[8.5/9] Fixing sparse folder structure for 3DGS..."

# Important fix:
# image_undistorter writes undistorted sparse files into scene/sparse/
# but 3DGS train.py expects them in scene/sparse/0/.
# Move original mapper result to distorted/sparse/0,
# and move undistorted files into sparse/0.
if [ -f "$SCENE_PATH/sparse/cameras.bin" ]; then
    mkdir -p "$SCENE_PATH/distorted/sparse"

    if [ -d "$SCENE_PATH/sparse/0" ]; then
        rm -rf "$SCENE_PATH/distorted/sparse/0"
        mv "$SCENE_PATH/sparse/0" "$SCENE_PATH/distorted/sparse/0"
    fi

    mkdir -p "$SCENE_PATH/sparse/0"
    mv "$SCENE_PATH/sparse/cameras.bin" "$SCENE_PATH/sparse/0/"
    mv "$SCENE_PATH/sparse/images.bin" "$SCENE_PATH/sparse/0/"
    mv "$SCENE_PATH/sparse/points3D.bin" "$SCENE_PATH/sparse/0/"
else
    echo "WARNING: Did not find $SCENE_PATH/sparse/cameras.bin"
    echo "Maybe image_undistorter already wrote to sparse/0. Continuing..."
fi

if [ ! -f "$SCENE_PATH/sparse/0/cameras.bin" ]; then
    echo "ERROR: Cannot find undistorted sparse/0/cameras.bin"
    exit 1
fi

echo ""
echo "[9/9] Running 3DGS training..."

cd "$GS_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting

if [ -n "$RESOLUTION" ]; then
    echo "Training with resolution parameter: -r $RESOLUTION"
    python train.py -s "$SCENE_PATH" -m "$OUTPUT_PATH" -r "$RESOLUTION"
else
    echo "Training with default resolution behavior."
    python train.py -s "$SCENE_PATH" -m "$OUTPUT_PATH"
fi

echo ""
echo "[Final] Compressing output folder..."

cd "$WORKDIR"
zip -qr "${SCENE_NAME}_output.zip" "${SCENE_NAME}_output"

echo "============================================================"
echo "All done."
echo ""
echo "Download this file from JupyterLab:"
echo "$OUTPUT_ZIP"
echo ""
echo "Final 3DGS ply:"
echo "$OUTPUT_PATH/point_cloud/iteration_30000/point_cloud.ply"
echo "============================================================"
