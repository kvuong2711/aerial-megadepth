# Aerial-MegaDepth

If you only want to check out the sample data, please skip to the [Sample Data](#sample-data) section.

If you want to generate the dataset on your own, please follow the instructions in the [Generating data from scratch](#generating-data-from-scratch) section.

## Sample Data
Since we do not own the images from Google Earth and MegaDepth, we cannot provide the full dataset. Instead, we provide a sample of the data (0001 in MegaDepth) to illustrate the structure and format of the dataset.

### Download Sample Data via CLI
Users can access the dataset using [AWS CLI](https://aws.amazon.com/cli/). Sample data can be downloaded using the following commands:

```
aws s3 sync s3://aerial-megadepth/scene_data/0001 ./0001
```

### Directory Structure of Sample Data
Here's the directory structure of the sample data:

```text
megadepth_aerial_tv/
└── data/
    └── 0001/
        └── sfm_output_localization/
            └── sfm_superpoint+superglue/
                ...
                └── localized_dense_metric/
                    ├── images/           # RGB images (contains Google Earth and MegaDepth images)
                    ├── depths/           # Depth maps
                    └── sparse-txt/       # COLMAP reconstruction (poses + intrinsics)
```


## Generating data from scratch

### Preparing pseudo-synthetic data
We provide the poses (in the form of .esp files) that you can use to render the images. The `.esp` file can be directly loaded into https://earth.google.com/studio/. Then the sequence of images can be rendered by the tool.


### 📁 Pseudo-Synthetic Directory Structure

The dataset is organized by scene ID. For each scene, we provide the original annotation file, raw downloads, and processed outputs:

```text
megadepth_aerial_tv/
├── geojsons/                          # Scene-level annotation files (.esp format)
│   ├── 0001/
│   │   └── 0001.esp
│   ├── 0002/
│   │   └── 0002.esp
│   └── ...
│
├── downloaded_data/                  # Raw downloaded footage and metadata
│   ├── 0001.json                     # Raw metadata (e.g., GPS, timestamps)
│   ├── 0001.mp4                      # Raw aerial footage for scene 0001
│   ├── 0002.json
│   ├── 0002.mp4
│   └── ...
│
├── data/                             # Processed outputs per scene
│   ├── 0001/
│   │   ├── 0001.json                 # Processed metadata (aligned to frames)
│   │   └── footage/
│   │       ├── frame_000000.jpeg    # Extracted video frames
│   │       ├── frame_000001.jpeg
│   │       └── ...
│   ├── 0002/
│   │   ├── 0002.json
│   │   └── footage/
│   │       ├── frame_000000.jpeg
│   │       └── ...
│   └── ...
```

### 📁 MegaDepth Processed Structure

We provide a set of preprocessed MegaDepth scenes containing camera parameters, depth maps, and aligned images. Each scene ID (e.g., `0001`) corresponds to a processed MegaDepth scene with the following structure:

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


### 📁 Data Generation

After running the data generation pipeline, outputs are saved under each scene directory as follows:

```text
megadepth_aerial_tv/
└── data/
    └── 0001/
        └── sfm_output_localization/
            └── sfm_superpoint+superglue/
                ...
                └── localized_dense_metric/
                    ├── images/           # RGB images
                    ├── depths/           # Depth maps
                    └── sparse-txt/       # COLMAP reconstruction (poses + intrinsics)
```

### TODO: provide pairs? or provide a script to generate pairs?

### TODO: dust3r's preprocess script to pack them into a single folder


