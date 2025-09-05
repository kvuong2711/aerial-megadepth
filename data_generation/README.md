# Aerial-MegaDepth Data Generation

📢 **NEWS (Sep 5, 2025): We have moved the data to HuggingFace to make it easier to download!**


We provide data and pipeline for generating our AerialMegaDepth dataset using Google Earth and MegaDepth. It includes a minimal example as well as instructions for generating your own data from scratch.

## Table of Contents
- [🗂️ Full Dataset Request](#%EF%B8%8F-full-dataset-request)
- [📦 Sample Data](#-sample-data)  
  - [Download via CLI](#download-via-cli)  
  - [Sample Data Structure](#sample-data-structure)  
- [🛠️ Generating Data from Scratch](#️-generating-data-from-scratch)  
  - [0️⃣ Prerequisites](#0️⃣-prerequisites)  
  - [1️⃣ Generating Pseudo-Synthetic Data from Google Earth Studio](#1️⃣-generating-pseudo-synthetic-data-from-google-earth-studio)  
  - [2️⃣ Registering to MegaDepth](#2️⃣-registering-to-megadepth)   
- [Prepare Data for Training DUSt3R/MASt3R](#prepare-data-for-training-dust3rmast3r)  
- [License](#license)


## 💡 Before you start...
For the following commands, `/mnt/slarge2/` is our local directory path where we store the data. You should replace it with the appropriate path on your machine.

If you run into issues preparing the dataset or are working on a research project that could benefit from our training data (particularly for academic use), feel free to reach out to me via [email](mailto:kvuong@andrew.cmu.edu). I'll do my best to help!


## 🗂️ Full Dataset Download
<!-- If you are interested in accessing the full dataset, please fill out this [Access Request Form](https://forms.gle/FrrverRecTFEwZKP9) and we will get back to you as soon as possible. Please note that the data is only available for non-commercial research purposes and strictly follows the [license](#license). -->

⚠️ **IMPORTANT**: The data is only available for **non-commercial research purposes and strictly follows the [license](#license).**

We are hosting the data on HuggingFace (https://huggingface.co/datasets/kvuong2711/aerialmegadepth), please submit the access request on HuggingFace. After being approved, you can install `huggingface_hub` and login using your HF's token:

```bash
pip install -U "huggingface_hub[cli]" 

hf auth login  # enter your HuggingFace token
```

Then, download the data using the following command:

```bash
python download_data_hf.py --repo_id kvuong2711/aerialmegadepth --local_dir /mnt/slarge2/megadepth_aerial_data/data --allow_patterns "**.zip" --unzip --remove_zip
```

#### (Optional) ☀️🌙  Time-varying Google Earth images
In addition to the standard data, we also provide Google Earth renderings at two alternative times of day (e.g., day/night). During training, these variants can be randomly used to improve robustness to lighting changes. While we have observed that the lighting model in Google Earth Studio is not photorealistic, as a form of data augmentation it may still offer slight improvements in scenarios with significant lighting variation.

As above, we are hosting the data on HuggingFace (https://huggingface.co/datasets/kvuong2711/aerialmegadepth-tv), please submit the access request on HuggingFace. After approval, you can download this data:

```bash
python download_data_hf.py --repo_id kvuong2711/aerialmegadepth-tv --local_dir /mnt/slarge2/megadepth_aerial_timevarying_data/data --allow_patterns "**.zip" --unzip --remove_zip --create_subdirs
```


## 📦 Sample Data

We provide a sample scene (`0001`) to illustrate the format and structure of the dataset. You can download it directly from Google Drive by CLI using [gdown](https://github.com/wkentaro/gdown). We provide a simple script [download_data.py](download_data.py) to download the data and unzip it.

### Download via CLI

You can use [gdown](https://github.com/wkentaro/gdown) to download the sample data (install using `pip install gdown`):

```bash
python download_data.py --output_path /mnt/slarge2/megadepth_aerial_data/data --gdrive_link "https://drive.google.com/open?id=1o8KGGicgMwp3ZK5FR5pRZIbHTwfndyiS" --unzip --remove_zip
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

1. [Generating Pseudo-Synthetic Data](#1-generating-pseudo-synthetic-data-from-google-earth-studio)  
2. [Registering to MegaDepth](#2-registering-to-megadepth)

### 0️⃣ Prerequisites
We provided a `.npz` file containing a list of scenes and images from MegaDepth in `datasets_preprocess/megadepth_image_list.npz`. These images will be registered to the corresponding pseudo-synthetic data.

### 1️⃣ Generating Pseudo-Synthetic Data from Google Earth Studio

This stage creates video frames and camera metadata using Google Earth Studio.

#### Step 1: Render Using Google Earth Studio

Each scene comes with pre-defined camera parameters in `.esp` format. You can download all `.esp` files using:

```bash
python download_data.py --output_path /mnt/slarge2/megadepth_aerial_data/ --gdrive_link "https://drive.google.com/open?id=1V-8ISc3OP7eZTpD2phcvB0O65m8ndnjs" --unzip --remove_zip
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

> 💡 Note: This step currently requires manual interaction with Google Earth Studio, which can be inconvenient. We actively welcome PRs or discussions that explore ways to automate or streamline this step!

#### Step 2: Extract Images & Metadata

Use the provided script to extract frames from each `.mp4` video and also extract camera metadata from the corresponding `.json` file:

```bash
python datasets_preprocess/preprocess_ge.py \
    --data_root /mnt/slarge2/megadepth_aerial_data \
    --scene_list ./datasets_preprocess/megadepth_image_list.npz
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
    --megadepth_image_list ./datasets_preprocess/megadepth_image_list.npz \
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


#### Step 2: Run the Data Registration Pipeline
※ Dependencies can be installed following the instructions in the [hloc repository](https://github.com/cvg/Hierarchical-Localization).


With both pseudo-synthetic frames and preprocessed MegaDepth data prepared, run the localization and reconstruction pipeline using:

```bash
python do_colmap_localization.py \
    --root_dir /mnt/slarge2/megadepth_aerial_data/data \
    --megadepth_dir /mnt/slarge2/megadepth_processed/ \
    --megadepth_image_list ./datasets_preprocess/megadepth_image_list.npz
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

## Prepare Data for Training DUSt3R/MASt3R
- We provide the precomputed pairs for training DUSt3R/MASt3R. First, download the precomputed pairs:

    ```bash
    python download_data.py --output_path ./ --gdrive_link "https://drive.google.com/open?id=19mGncey98Ci4lWuvl3BZ2_7I7_eKV6MS" --unzip --remove_zip
    ```

    Then, use the following script to preprocess the data to be in the format compatible with DUSt3R or MASt3R training:

    ```bash
    python datasets_preprocess/preprocess_aerialmegadepth.py \
        --megadepth_aerial_dir /mnt/slarge2/megadepth_aerial_data/data \
        --precomputed_pairs ./data_splits/aerial_megadepth_train_part1.npz \
        --output_dir /mnt/slarge2/megadepth_aerial_processed
    ```

- (Optional) Since ground-truth depth maps computed from MVS often contain outliers, especially around object boundaries (e.g., between buildings and the sky), during training we apply semantic segmentation to detect sky regions and mask out their depth values. We provide precomputed semantic segmentation masks generated using [InternImage](https://github.com/OpenGVLab/InternImage). Note that the effect of this filtering step was not thoroughly analyzed in our paper. You can download the segmentation masks using:

    ```bash
    python download_data.py --output_path /mnt/slarge2/ --gdrive_link "https://drive.google.com/open?id=1j6wDFO1r5psNdBM8HIvk_cYGZa3Kde97" --unzip --remove_zip
    ```

    We provided a sample DUSt3R-based dataloader for training on AerialMegaDepth in [`misc/megadepth_aerial.py`](misc/megadepth_aerial.py).

- Finetuning details: we follow the original DUSt3R/MASt3R training settings by randomly sampling 100K pairs per epoch to train (see [here](https://github.com/naver/dust3r?tab=readme-ov-file#our-hyperparameters)), except to avoid overfitting, we use a smaller learning rate of `1e-5` for DUSt3R and `3e-5` for MASt3R.


## License
Google Earth data belong to [Google](https://www.google.com/earth/studio/faq/) and is available for non-commercial research purposes only. For full information, please refer to their [TOS](https://earthengine.google.com/terms/). All data derived from Google is owned by Google, while other parts of the dataset from MegaDepth are subject to their original licensing terms.