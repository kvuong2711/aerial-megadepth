from pathlib import Path
import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation
from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    pairs_from_poses,
    triangulation,
    pairs_from_exhaustive,
)
import subprocess
import open3d as o3d
import argparse
from hloc.utils.read_write_model import Camera, Image, Point3D, read_model, write_model, rotmat2qvec, qvec2rotmat
from hloc.utils.read_write_dense import read_array
import pycolmap
import matplotlib.pyplot as plt
from hloc.visualization import plot_images, read_image
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc import localization_colmap
from hloc.localization_colmap import run_command
from ges_utils import json_to_empty_colmap_model
from tqdm import tqdm
import os.path as osp
import h5py


def build_queries_cam(query_images_folder, queries_txt):
    queries = []
    # Find all jpg images
    rgb_images = sorted(list(glob.glob(os.path.join(query_images_folder, '*.jpg'))))

    for i, img in enumerate(rgb_images):
        # get the camera intrinsics (add .npz to the image name)
        basename = os.path.basename(img)
        cam_file = os.path.join(query_images_folder, basename[0:-4] + '.npz')
        # get the intrinsics
        intrinsics = np.load(cam_file)['intrinsics']
        cam_model = 'SIMPLE_PINHOLE'
        f = intrinsics[0, 0]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        width, height = (cx + 0.5) * 2, (cy + 0.5) * 2
        width, height = int(width), int(height)

        queries.append((basename, cam_model, width, height, f, cx, cy))

    print('Number of query images:', len(queries))
    print('>>> Writing queries to file:', queries_txt)
    with open(queries_txt, "w") as f:
        for query in queries:
            f.write(f"{query[0]} {query[1]} {query[2]} {query[3]} {query[4]} {query[5]} {query[6]}\n")

    # Only take the query image list as returning value
    query_list = []
    for query in queries:
        query_list.append(query[0])

    return query_list


def megadepth_to_colmap_model(root_dir, ref_sfm_empty):
    # create colmap images
    colmap_images = {}
    colmap_cameras = {}

    rgb_images = sorted(list(glob.glob(os.path.join(root_dir, '*.jpg'))))

    cam_id_start = 1000
    image_id_start = 1000

    # Loop through the images of the reconstruction
    for idx, image_full_name in enumerate(rgb_images):
        image_name = os.path.basename(image_full_name)
        # get the camera intrinsics (add .npz to the image name)
        basename = image_name
        # Load the pose + intrinsics
        cam_file = os.path.join(root_dir, basename[0:-4] + '.npz')
        # get the intrinsics
        intrinsics = np.load(cam_file)['intrinsics']
        cam_model = 'SIMPLE_PINHOLE'
        f = intrinsics[0, 0]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        width, height = (cx + 0.5) * 2, (cy + 0.5) * 2
        width, height = int(width), int(height)
        ret_camera = pycolmap.Camera(model=cam_model, width=width, height=height, params=np.array([f, cx, cy]))
        # get the pose
        P_T_C = np.load(cam_file)['cam2world']

        # create the same camera in colmap
        cam_id = cam_id_start + idx
        curr_cam = Camera(
            id=cam_id,
            model="SIMPLE_PINHOLE",
            width=ret_camera.width,
            height=ret_camera.height,
            params=ret_camera.params
        )
        colmap_cameras[cam_id] = curr_cam

        # create the image, with the same image id
        image_id = image_id_start + idx
        w2c = np.linalg.inv(P_T_C)
        colmap_images[image_id] = Image(
            id=image_id,
            qvec=rotmat2qvec(w2c[0:3, 0:3]),
            tvec=w2c[0:3, 3],
            camera_id=cam_id,
            name=image_name,
            xys=np.empty([0, 2]),
            point3D_ids=np.empty(0),
        )

    # create COLMAP points3D
    points3D = {}
    print(">>> CAMERAS: ", len(colmap_cameras))
    print(">>> IMAGES: ", len(colmap_images))
    print(">>> POINTS3D: ", len(points3D))
    os.makedirs(ref_sfm_empty, exist_ok=True)
    # write_model(cameras, images, points3D, ref_sfm_empty, ".txt")
    write_model(colmap_cameras, colmap_images, points3D, ref_sfm_empty, ".bin")



def run_hloc_triangulation(json_file, root_dir, output_dir):
    # First, convert the JSON file to the COLMAP
    output_dir.mkdir(exist_ok=True, parents=True)
    ref_sfm_empty = output_dir / "sfm_reference_empty"
    json_to_empty_colmap_model(json_file, ref_sfm_empty, max_num_images=None)

    num_ref_pairs = 50
    ref_pairs = output_dir / f"pairs-db-dist{num_ref_pairs}.txt"
    # Match reference images that are spatially close
    pairs_from_poses.main(ref_sfm_empty, ref_pairs, num_ref_pairs, rotation_threshold=75)

    # Extract, match, and triangulate the reference SfM model
    fconf = extract_features.confs["superpoint_max"]
    mconf = match_features.confs["superglue"]
    ref_sfm = output_dir / "sfm_superpoint+superglue"

    ref_images = root_dir / "footage"

    # let's get the unique images 
    with open(ref_pairs, "r") as f:
        lines = f.readlines()
        ref_images_list = set()
        for line in lines:
            ref_images_list.add(line.split()[0])
            ref_images_list.add(line.split()[1])

    ref_images_list = list(ref_images_list)

    # Extract features and match the reference images
    ffile = extract_features.main(conf=fconf, image_dir=ref_images, export_dir=output_dir, image_list=ref_images_list)
    mfile = match_features.main(conf=mconf, pairs=ref_pairs, features=fconf["output"], export_dir=output_dir)
    print('Features file:', ffile)
    print('Matches file:', mfile)

    # Run triangulation
    triangulation.main(ref_sfm, ref_sfm_empty, ref_images, ref_pairs, ffile, mfile)


def run_colmap_localization(root_dir, output_dir, query_images):
    # Make sure this configuration matches one in run_hloc_triangulation()
    fconf = extract_features.confs["superpoint_max"]
    mconf = match_features.confs["superglue"]
    ref_sfm = output_dir / "sfm_superpoint+superglue" 

    ref_images = root_dir / "footage"
    ffile = Path(output_dir, fconf["output"] + ".h5")  # this is db features, defined in extract_features.py

    # Load existing sparse pseudo-synthetic reconstruction
    model = pycolmap.Reconstruction(ref_sfm)

    # In this context, query images are MegaDepth images, and reference images are pseudo-synthetic images
    queries_txt = output_dir / "queries.txt"
    query_list = build_queries_cam(query_images_folder=query_images, queries_txt=queries_txt)
    
    # Define output localization paths
    loc_output_dir = output_dir / "localization"

    # Extract features and match the query image to the reference images.
    ffile_q = extract_features.main(conf=fconf, 
                                    image_dir=query_images, 
                                    image_list=query_list, 
                                    export_dir=loc_output_dir,
                                    overwrite=False)
    print('Features (query) file:', ffile_q)

    # For every query image, we need to find the reference images that are similar to it
    # Either we do exhaustive matching or use global features like netvlad/etc.
    # references_registered = [model.images[i].name for i in model.reg_image_ids()]
    # pairs_from_exhaustive.main(loc_pairs, image_list=query_list, ref_list=references_registered)

    references_registered = [model.images[i].name for i in model.reg_image_ids()]
    retrieval_conf = extract_features.confs["netvlad"]
    # retrieval_conf = extract_features.confs["eigenplaces"]

    db_descriptors = Path(loc_output_dir, 'db_' + retrieval_conf["output"] + '.h5')
    extract_features.main(conf=retrieval_conf, 
                            image_dir=ref_images, 
                            image_list=references_registered,
                            feature_path=db_descriptors,
                            export_dir=loc_output_dir)
    query_descriptors = Path(loc_output_dir, 'query_' + retrieval_conf["output"] + '.h5')
    extract_features.main(conf=retrieval_conf,
                            image_dir=query_images,
                            image_list=query_list,
                            feature_path=query_descriptors,
                            export_dir=loc_output_dir)

    num_loc_pairs = 50
    loc_pairs = loc_output_dir / f"pairs-query-dist{num_loc_pairs}.txt"

    pairs_from_retrieval.main(
        descriptors=query_descriptors,
        output=loc_pairs,
        num_matched=num_loc_pairs,
        query_list=query_list,
        db_list=references_registered,
        db_model=ref_sfm,
        db_descriptors=db_descriptors
    )

    # Match the query image to the reference images.
    mfile_q = loc_output_dir / f"matches-superglue_{loc_pairs.stem}.h5"
    match_features.main(conf=mconf, pairs=loc_pairs, 
                        features=ffile_q, 
                        features_ref=ffile,
                        matches=mfile_q, 
                        overwrite=False)
    
    """
    Next, we need to also establish matches/corresondences among the query (MegaDepth) images
    """
    # First, create empty megadepth model with poses
    ref_megadepth_sfm_empty = output_dir / "sfm_megadepth_empty"
    megadepth_to_colmap_model(query_images, ref_megadepth_sfm_empty)

    num_ref_pairs = 20
    ref_megadepth_pairs = output_dir / f"pairs-among-query-dist{num_ref_pairs}.txt"
    # Match reference images that are spatially close.
    pairs_from_poses.main(ref_megadepth_sfm_empty, ref_megadepth_pairs, num_ref_pairs, rotation_threshold=75)

    # Match features
    match_features.main(conf=mconf, 
                        pairs=ref_megadepth_pairs, 
                        features=ffile_q, 
                        features_ref=ffile_q,
                        matches=mfile_q, 
                        overwrite=False)
    
    # Merge two pairs txt files, one is among query images, the other is between query and reference images
    loc_pairs_merged = loc_output_dir / f"pairs-localization-merged.txt"
    with open(loc_pairs, 'r') as f1, open(ref_megadepth_pairs, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        with open(loc_pairs_merged, 'w') as f:
            f.writelines(lines1)
            # add a new line
            f.write('\n')
            f.writelines(lines2)

    # Insert images into colmap to do the localization
    localization_colmap.main(
        sfm_dir=ref_sfm,
        db_image_dir=ref_images,
        query_image_dir=query_images,
        pairs=loc_pairs_merged,
        features=ffile_q,
        matches=mfile_q,
        verbose=True
    )
    

def run_mvs(sparse_sfm, dense_sfm, image_dir, gpu_idx=0):
    colmap_path = 'colmap'
    cmd = [
        str(colmap_path), 'image_undistorter',
        '--image_path', str(image_dir),
        '--input_path', str(sparse_sfm),
        '--output_path', str(dense_sfm)]
    run_command(cmd, True)

    cmd = [
        str(colmap_path), 'patch_match_stereo',
        '--workspace_path', str(dense_sfm),
        '--PatchMatchStereo.cache_size', '32',
        '--PatchMatchStereo.gpu_index', str(gpu_idx),
        '--PatchMatchStereo.max_image_size', '2000',
        ]
    run_command(cmd, True)

    cmd = [
        str(colmap_path), 'stereo_fusion',
        '--workspace_path', str(dense_sfm),
        '--output_path', str(dense_sfm / 'fused.ply')]
    run_command(cmd, True)



def depth_to_h5(depth_map, file_name, depth_h5_dir):
    h5_path = os.path.join(depth_h5_dir, '%s.h5' % os.path.splitext(file_name)[0])

    with h5py.File(h5_path, "w") as f:
        dest = f.create_dataset("depth", data=depth_map, compression='gzip', compression_opts=9)
        dest.attrs["description"] = "Depth map of the image"


def postprocess_scene(base_undistorted_sfm_path, scene_id):
    print('>>> Postprocessing scene:', scene_id)

    undistorted_sparse_path = os.path.join(
        base_undistorted_sfm_path, scene_id, 'sfm_output_localization', 'sfm_superpoint+superglue', 'localized_dense_metric', 'sparse'
    )
    images_path = os.path.join(
        base_undistorted_sfm_path, scene_id, 'sfm_output_localization', 'sfm_superpoint+superglue', 'localized_dense_metric', 'images'
    )
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path {images_path} does not exist.")

    depths_path = os.path.join(
        base_undistorted_sfm_path, scene_id, 'sfm_output_localization', 'sfm_superpoint+superglue', 'localized_dense_metric', 'stereo', 'depth_maps'
    )
    if not os.path.exists(depths_path):
        raise FileNotFoundError(f"Depth maps path {depths_path} does not exist.")
    
    # Transform the reconstruction to raw text format.
    sparse_txt_path = os.path.join(base_undistorted_sfm_path, scene_id, 'sfm_output_localization', 'sfm_superpoint+superglue', 'localized_dense_metric', 'sparse-txt')
    os.makedirs(sparse_txt_path, exist_ok=True)

    cmd = [
        'colmap', 'model_converter',
        '--input_path', undistorted_sparse_path,
        '--output_path', sparse_txt_path, 
        '--output_type', 'TXT'
    ]
    run_command(cmd, True)

    # Output the depth maps as h5 files
    depths_output = os.path.join(base_undistorted_sfm_path, scene_id, 'sfm_output_localization', 'sfm_superpoint+superglue', 'localized_dense_metric', 'depths')
    os.makedirs(depths_output, exist_ok=True)

    # For every image, find the corresponding depth map and convert it to h5 format
    image_name_list = sorted(os.listdir(images_path))
    for image_name in tqdm(image_name_list):
        depth_name = image_name + '.geometric.bin'
        depth_path = os.path.join(depths_path, depth_name)
        if not os.path.exists(depth_path):
            print(f"Depth map {depth_path} does not exist.")
            continue

        # First, let's load the depth map
        depth_map = read_array(depth_path)

        # Save the depth map as an h5 file
        depth_to_h5(depth_map, image_name, depths_output)



def parse_args():
    parser = argparse.ArgumentParser(description="Convert GES JSON to COLMAP format")
    parser.add_argument("--root_dir", type=Path, required=True, help='Path to the root directory containing the scene folders')
    parser.add_argument("--megadepth_dir", type=Path, required=True, help='Path to the processed MegaDepth dataset directory')
    parser.add_argument("--megadepth_image_list", type=Path, required=True, help='Path to .npz file listing scene names to process')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    init_args = parse_args()
    root_dir = init_args.root_dir
    megadepth_dir = init_args.megadepth_dir
    megadepth_image_list = init_args.megadepth_image_list

    # Load the megadepth image list
    megadepth_data_dict = np.load(megadepth_image_list, allow_pickle=True)['data'].item()
    synthetic_scenes = list(megadepth_data_dict.keys())
    synthetic_scenes = [scene[0] for scene in synthetic_scenes]  # (scene_id, subscene_id)
    print('Loading a total of {} scenes'.format(len(synthetic_scenes)))
    
    for scene_number in synthetic_scenes:
        # Check if the scene folder exists
        data_folder_dir = root_dir / f'{scene_number}'
        json_file = data_folder_dir / f'{scene_number}.json'
        assert json_file.exists(), f"JSON file {json_file} does not exist"

        output_dir = data_folder_dir / "sfm_output_localization"

        # First, triangulate the sparse pseudo-synthetic scene using GT poses with hloc
        print(f">>>>> Triangulating the sparse pseudo-synthetic scene using GT poses with hloc")
        run_hloc_triangulation(json_file, data_folder_dir, output_dir)

        # Next, localize MegaDepth images into the sparse pseudo-synthetic scene
        print(f"Running localization for {scene_number}")
        sfm_db = output_dir / "sfm_superpoint+superglue" / 'images.bin'
        # Making sure the sparse pseudo-synthetic scene is successfully triangulated
        if not sfm_db.exists():
            print(f">>>>> Triangulation for {scene_number} not exists yet. Skipping localization.")
            continue
        query_images_folder_scene = Path(f'{megadepth_dir}/{scene_number}/0')
        run_colmap_localization(data_folder_dir, output_dir, query_images_folder_scene)

        # Now, run MVS
        sparse_sfm = output_dir / "sfm_superpoint+superglue" / "localized_model_mapper_metric"
        dense_sfm = output_dir / "sfm_superpoint+superglue" / "localized_dense_metric"
        image_dir = output_dir / "sfm_superpoint+superglue" / "images"

        run_mvs(sparse_sfm, dense_sfm, image_dir, gpu_idx=0)

        # Postprocess the scene (re-organize and convert depth maps to h5 format)
        postprocess_scene(output_dir, scene_number)
