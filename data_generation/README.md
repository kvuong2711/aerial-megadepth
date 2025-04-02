# Aerial-MegaDepth Data Generation

This repository provides a dataset and pipeline for generating pseudo-synthetic multi-view image collections using Google Earth and MegaDepth. It includes pre-rendered samples as well as instructions for generating your own data from scratch.

> **Disclaimer**: Due to licensing restrictions, we do not redistribute Google Earth or MegaDepth images in bulk. Instead, we provide a minimal example and tools to reproduce the full dataset.

## 📦 Sample Data

We provide a sample scene (`0001`) to illustrate the format and structure of the dataset. You can download it directly using the AWS CLI.

### Download via CLI

You can use [AWS CLI](https://aws.amazon.com/cli/) to download the sample data:

```bash
mkdir -p /mnt/slarge2/megadepth_aerial_data/data
aws s3 sync s3://aerial-megadepth/full_data/0001 /mnt/slarge2/megadepth_aerial_data/data/0001
```
This command will download the sample scene data to `/mnt/slarge2/megadepth_aerial_data/data/0001`.

### Sample Data Structure

```
megadepth_aerial_data/
└── data/
    └── 0001/
        └── sfm_output_localization/
            └── sfm_superpoint+superglue/
                └── localized_dense_metric/
                    ├── images/           # RGB images (Google Earth & MegaDepth)
                    ├── depths/           # Depth maps
                    └── sparse-txt/       # COLMAP reconstruction files
```

## 🛠️ Generating Data from Scratch

The full pipeline involves two stages:

1. [Generating Pseudo-Synthetic Data](#1-generating-pseudo-synthetic-data)  
2. [Registering to MegaDepth](#2-registering-to-megadepth)

### 0️⃣ Prerequisites
We provided a `.npz` file containing a list of images from MegaDepth in `datasets_preprocess/megadepth_image_list_v0.npz`. These images will be registered to the pseudo-synthetic data.

### 1️⃣ Generating Pseudo-Synthetic Data from Google Earth Studio

This stage creates video frames and camera metadata using Google Earth Studio.

#### Step 1: Render Using Google Earth Studio

Each scene comes with pre-defined camera parameters in `.esp` format. You can download all `.esp` files using:

```bash
aws s3 sync s3://aerial-megadepth/geojsons /mnt/slarge2/megadepth_aerial_data/geojsons
```

Directory structure:

```
megadepth_aerial_data/
└── geojsons/
    ├── 0001/
    │   └── 0001.esp
    ├── 0002/
    │   └── 0002.esp
    └── ...
```

To render the pseudo-synthetic sequence:

1. Open [Google Earth Studio](https://earth.google.com/studio/)
2. Import a `.esp` file via **File → Import → Earth Studio Project**
3. Go to **Render** and export:
   - **Video**: select **Cloud Rendering** to produce a `.mp4`
   - **Tracking Data**: enable **3D Camera Tracking (JSON)** with **Coordinate Space: Global**

Save the exported files to:

```
megadepth_aerial_data/
└── downloaded_data/
    ├── 0001.mp4     # Rendered video
    ├── 0001.json    # Camera metadata (pose, intrinsics, timestamps)
    └── ...
```

> 💡 **Note:** This step requires manual interaction with Google Earth Studio which is a bit inconvenient. Therefore, we actively welcome [PRs](https://github.com/your-repo-url) or discussions that help automate this step or streamline rendering workflows.

#### Step 2: Extract Frames & Align Metadata

Use the provided script to extract frames from each `.mp4` video and align them with camera metadata from the corresponding `.json` file:

```bash
python datasets_preprocess/preprocess_ge.py \
    --data_root /mnt/slarge2/megadepth_aerial_data \
    --scene_list ./datasets_preprocess/megadepth_image_list_v0.npz
```

This will generate per-scene folders with extracted frames and frame-aligned metadata:

```
megadepth_aerial_data/
└── data/
    ├── 0001/
    │   ├── 0001.json               # Aligned metadata (pose, intrinsics, timestamps)
    │   └── footage/
    │       ├── 0001_000.jpeg       # Extracted video frames
    │       ├── 0001_001.jpeg
    │       └── ...
    └── ...
```


### 2️⃣ Registering to MegaDepth

Once pseudo-synthetic images are generated, the next step is to localize them within a MegaDepth scene and reconstruct the scene geometry.

#### Step 1: Prepare MegaDepth Images

First, download the [MegaDepth dataset](https://www.cs.cornell.edu/projects/megadepth/) by following their instructions. After downloading, your dataset root (e.g., `/mnt/slarge/megadepth_original`) should contain the folders `MegaDepth_v1_SfM` and `phoenix`.

Then, use the provided preprocessing script to extract RGB images, depth maps, and camera parameters for each scene:

```bash
python datasets_preprocess/preprocess_megadepth.py \
    --megadepth_dir /mnt/slarge/megadepth_original/MegaDepth_v1_SfM \
    --megadepth_image_list ./datasets_preprocess/megadepth_image_list_v0.npz \
    --output_dir /mnt/slarge2/megadepth_processed
```

This will generate processed outputs with the following structure:

```text
megadepth_processed/
├── 0001/
│   └── 0/
│       ├── 5008984_74a994ce1c_o.jpg.jpg    # RGB image
│       ├── 5008984_74a994ce1c_o.jpg.exr    # Depth map (EXR format)
│       ├── 5008984_74a994ce1c_o.jpg.npz    # Camera pose + intrinsics
│       └── ...
├── 0002/
│   └── 0/
│       └── ...
└── ...
```

Each `.jpg` file corresponds to a view and is paired with:
- a `.npz` file containing camera intrinsics and extrinsics
- a `.exr` file containing a depth map in metric scale


#### Step 2: Run the Data Generation Pipeline

With both pseudo-synthetic frames and preprocessed MegaDepth data prepared, run the localization and reconstruction pipeline using:

```bash
python do_colmap_localization.py \
    --root_dir /mnt/slarge2/megadepth_aerial_data/data \
    --megadepth_dir /mnt/slarge2/megadepth_processed/ \
    --megadepth_image_list ./datasets_preprocess/megadepth_image_list_v0.npz
```

The output is saved per scene as:

```
megadepth_aerial_data/
└── data/
    └── 0001/
        └── sfm_output_localization/
            └── sfm_superpoint+superglue/
                └── localized_dense_metric/
                    ├── images/           # Registered RGB images
                    ├── depths/           # MVS depth maps
                    └── sparse-txt/       # COLMAP poses + intrinsics (text format)
```
