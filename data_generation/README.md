# Aerial-MegaDepth Data Generation

This repository provides a dataset and pipeline for generating pseudo-synthetic multi-view image collections using Google Earth and MegaDepth. It includes pre-rendered samples as well as instructions for generating your own data from scratch.

> **Disclaimer**: Due to licensing restrictions, we do not redistribute Google Earth or MegaDepth images in bulk. Instead, we provide a minimal example and tools to reproduce the full dataset.

## 📦 Sample Data

We provide a sample scene (`0001`) to illustrate the format and structure of the dataset. You can download it directly using the AWS CLI.

### Download via CLI

First, install the [AWS CLI](https://aws.amazon.com/cli/) if you haven’t already. Then run:

```bash
mkdir -p megadepth_aerial_data/data
aws s3 sync s3://aerial-megadepth/full_data/0001 megadepth_aerial_data/data/0001
```
This command will download the sample scene data to `megadepth_aerial_data/data/0001`.

### Sample Data Structure

```
megadepth_aerial_tv/
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

### 1️⃣ Generating Pseudo-Synthetic Data from Google Earth Studio

This stage creates video frames and camera metadata using Google Earth Studio.

#### Step 1: Render Using Google Earth Studio

Each scene comes with pre-defined camera parameters in `.esp` format. You can download all `.esp` files using:

```bash
aws s3 sync s3://aerial-megadepth/geojsons ./megadepth_aerial_tv/geojsons
```

Directory structure:

```
megadepth_aerial_tv/
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
megadepth_aerial_tv/
└── downloaded_data/
    ├── 0001.mp4     # Rendered video
    ├── 0001.json    # Camera metadata (pose, intrinsics, timestamps)
    └── ...
```

> 💡 **Note:** This step requires manual interaction with Google Earth Studio which is a bit inconvenient. Therefore, we actively welcome [pull requests](https://github.com/your-repo-url) or discussions that help automate this step or streamline rendering workflows.

#### Step 2: Extract Frames & Align Metadata

Use a preprocessing script to extract video frames and organize the data structure like this:
#### TODO: add the script

```
megadepth_aerial_tv/
└── data/
    ├── 0001/
    │   ├── 0001.json               # camera parameters
    │   └── footage/
    │       ├── frame_000000.jpeg
    │       ├── frame_000001.jpeg
    │       └── ...
    └── ...
```
---

### 2️⃣ Registering to MegaDepth

Once pseudo-synthetic images are generated, the next step is to localize them in a MegaDepth scene and reconstruct the geometry.

#### Step 1: Prepare MegaDepth Images

Use the provided preprocessing script to extract images, depths, and camera poses from a MegaDepth scene. The processed files are saved in the following format:

```text
megadepth_processed_extra/
├── 0001/
│   └── 0/
│       ├── 5008984_74a994ce1c_o.jpg.npz    # Camera pose + intrinsics
│       ├── 5008984_74a994ce1c_o.jpg.exr    # Depth map (EXR format)
│       ├── 5008984_74a994ce1c_o.jpg.jpg    # RGB image
│       └── ...
├── 0002/
│   └── 0/
│       └── ...
└── ...
```

Each `.jpg` file has a matching `.npz` (camera parameters) and `.exr` (depth map).

---

#### Step 2: Run the Data Generation Pipeline

With both pseudo-synthetic and MegaDepth data prepared, you can run the localization and reconstruction pipeline (e.g., SuperPoint + SuperGlue + COLMAP). The output is saved per scene as:

```text
megadepth_aerial_tv/
└── data/
    └── 0001/
        └── sfm_output_localization/
            └── sfm_superpoint+superglue/
                └── localized_dense_metric/
                    ├── images/           # Registered RGB images
                    ├── depths/           # Optional MVS depth maps
                    └── sparse-txt/       # COLMAP poses + intrinsics (text format)
```

---

Let me know if you'd like this split into a checklist or numbered steps, or if you'd like help formatting the CLI commands for running each stage.