#!/usr/bin/env python3
# Ported from DUSt3R
#
# --------------------------------------------------------
# Preprocessing code for the AerialMegaDepth dataset
# --------------------------------------------------------
import os
import os.path as osp
import collections
from tqdm import tqdm
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import h5py
import PIL.Image
from utils import parallel_threads, rescale_image_depthmap
import matplotlib.pyplot as plt
import shutil


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--megasynth_dir', required=True)
    parser.add_argument('--precomputed_pairs', required=True)
    parser.add_argument('--output_dir', default='data/megasynth_processed')
    return parser


def main(db_root, pairs_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # load all pairs
    data = np.load(pairs_path, allow_pickle=True)
    scenes = data['scenes']
    images = data['images']
    pairs = data['pairs']
    images_scene_name = data['images_scene_name']

    print('Loaded', len(pairs), 'pairs from', len(scenes), 'scenes')

    # let's add all images to the todo list (not just the ones in pairs)
    todo = collections.defaultdict(set)
    for image_id, image in enumerate(images):
        if image != None:
            scene = images_scene_name[image_id]
            scene_id = np.where(scenes == scene)[0][0]
            todo[scene_id].add(image_id)
        # else:
        #     print(image_id, image, images_scene_name[image_id])

    print('Number of unique images:', np.sum([len(v) for v in todo.values()]))

    # for each scene, load intrinsics and then parallel crops
    for scene, im_idxs in tqdm(todo.items(), desc='Overall'):
        scene = scenes[scene]
        out_dir = osp.join(output_dir, scene)
        
        os.makedirs(out_dir, exist_ok=True)

        # load all camera params
        _, pose_w2cam, intrinsics = _load_kpts_and_poses(db_root, scene, intrinsics=True)

        # # in_dir = osp.join(db_root, scene, 'dense' + subscene)
        in_dir = osp.join(os.path.dirname(db_root), scene, "sfm_output_localization", "sfm_superpoint+superglue", "localized_dense")
        args = [(in_dir, img, intrinsics[img], pose_w2cam[img], out_dir)
                for img in [images[im_id] for im_id in im_idxs]]
        
        # normal loop
        # for arg in args:
        #     resize_one_image(*arg)
        parallel_threads(resize_one_image, args, star_args=True, front_num=0, leave=False, desc=f'{scene}')


    # # for each scene, load intrinsics and then parallel crops
    # for scene, im_idxs in tqdm(todo.items(), desc='Overall'):
    #     scene, subscene = scenes[scene].split()
    #     out_dir = osp.join(output_dir, scene, subscene)
    #     os.makedirs(out_dir, exist_ok=True)

    #     # load all camera params
    #     _, pose_w2cam, intrinsics = _load_kpts_and_poses(db_root, scene, subscene, intrinsics=True)

    #     # in_dir = osp.join(db_root, scene, 'dense' + subscene)
    #     in_dir = osp.join(os.path.dirname(db_root), 'phoenix/S6/zl548/MegaDepth_v1/', scene, 'dense' + subscene)
    #     args = [(in_dir, img, intrinsics[img], pose_w2cam[img], out_dir)
    #             for img in [images[im_id] for im_id in im_idxs]]
    #     parallel_threads(resize_one_image, args, star_args=True, front_num=0, leave=False, desc=f'{scene}/{subscene}')

    # save pairs
    print('Done! prepared all pairs in', output_dir)


def resize_one_image(root, tag, K_pre_rectif, pose_w2cam, out_dir):
    # if osp.isfile(osp.join(out_dir, tag + '.npz')):
    #     return

    # # load image
    # img = cv2.cvtColor(cv2.imread(osp.join(root, 'images', tag), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # H, W = img.shape[:2]

    # H, W = 1080, 1920  # TODO: this is just for megasynth where image size is constant

    # load depth
    with h5py.File(osp.join(root, 'depths', osp.splitext(tag)[0] + '.h5'), 'r') as hd5:
        depthmap = np.asarray(hd5['depth'])

    # rectify = undistort the intrinsics
    imsize_pre, K_pre, distortion = K_pre_rectif
    # imsize_post = img.shape[1::-1]
    # K_post = cv2.getOptimalNewCameraMatrix(K_pre, distortion, imsize_pre, alpha=0,
    #                                        newImgSize=imsize_post, centerPrincipalPoint=True)[0]

    # # downscale
    # img_out, depthmap_out, intrinsics_out, R_in2out = _downscale_image(K_post, img, depthmap, resolution_out=(800, 600))
    # img_out, depthmap_out, intrinsics_out, R_in2out = PIL.Image.fromarray(img), depthmap, K_pre, np.eye(3)
    depthmap_out, intrinsics_out, R_in2out = depthmap, K_pre, np.eye(3)
    # let's set the depth too far away to 0.0
    MAX_DEPTH = 2000
    depthmap_out[depthmap_out > MAX_DEPTH] = 0.0
    # Load segmentation
    # seg = PIL.Image.open(osp.join(root, 'segmentation', osp.splitext(tag)[0] + '.png'))
    ext = osp.splitext(tag)[1]
    seg = PIL.Image.open(osp.join(root, 'segmentation', tag.replace(ext, '.png')))
    seg = np.array(seg)
    # mask out sky (2) and person (12)
    mask = (seg == 2) | (seg == 12)
    depthmap_out[mask] = 0.0
    # plt.imshow(mask)
    # plt.show()

    # write everything
    # img_out.save(osp.join(out_dir, tag + '.jpg'), quality=90)
    # copy the image to the output directory
    shutil.copy(osp.join(root, 'images', tag), osp.join(out_dir, tag + '.jpg'))
    cv2.imwrite(osp.join(out_dir, tag + '.exr'), depthmap_out)
    camout2world = np.linalg.inv(pose_w2cam)
    camout2world[:3, :3] = camout2world[:3, :3] @ R_in2out.T
    np.savez(osp.join(out_dir, tag + '.npz'), intrinsics=intrinsics_out, cam2world=camout2world)



    # # write everything
    # # img_out.save(osp.join(out_dir, tag + '.jpg'), quality=90)
    # cv2.imwrite(osp.join(out_dir, tag + '.exr'), depthmap_out)
    # camout2world = np.linalg.inv(pose_w2cam)
    # camout2world[:3, :3] = camout2world[:3, :3] @ R_in2out.T
    # np.savez(osp.join(out_dir, tag + '.npz'), intrinsics=intrinsics_out, cam2world=camout2world)


def _downscale_image(camera_intrinsics, image, depthmap, resolution_out=(512, 384)):
    H, W = image.shape[:2]
    resolution_out = sorted(resolution_out)[::+1 if W < H else -1]

    image, depthmap, intrinsics_out = rescale_image_depthmap(
        image, depthmap, camera_intrinsics, resolution_out, force=False)
    R_in2out = np.eye(3)

    return image, depthmap, intrinsics_out, R_in2out


def _load_kpts_and_poses(root, scene_id, z_only=False, intrinsics=False):
    if intrinsics:
        with open(os.path.join(root, scene_id, "sfm_output_localization", "sfm_superpoint+superglue", "localized_dense", 'sparse-txt', 'cameras.txt'), 'r') as f:
            raw = f.readlines()[3:]  # skip the header

        camera_intrinsics = {}
        for camera in raw:
            camera = camera.split(' ')
            width, height, focal, cx, cy = [float(elem) for elem in camera[2:]]
            K = np.eye(3)
            K[0, 0] = focal
            K[1, 1] = focal
            K[0, 2] = cx
            K[1, 2] = cy
            camera_intrinsics[int(camera[0])] = ((int(width), int(height)), K, (0, 0, 0, 0))

    with open(os.path.join(root, scene_id, "sfm_output_localization", "sfm_superpoint+superglue", "localized_dense", 'sparse-txt', 'images.txt'), 'r') as f:
        raw = f.read().splitlines()[4:]  # skip the header

    extract_pose = colmap_raw_pose_to_principal_axis if z_only else colmap_raw_pose_to_RT

    poses = {}
    points3D_idxs = {}
    camera = []

    for image, points in zip(raw[:: 2], raw[1:: 2]):
        image = image.split(' ')
        points = points.split(' ')

        image_id = image[-1]
        camera.append(int(image[-2]))

        # find the principal axis
        raw_pose = [float(elem) for elem in image[1: -2]]
        poses[image_id] = extract_pose(raw_pose)

        current_points3D_idxs = {int(i) for i in points[2:: 3] if i != '-1'}
        assert -1 not in current_points3D_idxs, bb()
        points3D_idxs[image_id] = current_points3D_idxs

    if intrinsics:
        image_intrinsics = {im_id: camera_intrinsics[cam] for im_id, cam in zip(poses, camera)}
        return points3D_idxs, poses, image_intrinsics
    else:
        return points3D_idxs, poses


def colmap_raw_pose_to_principal_axis(image_pose):
    qvec = image_pose[: 4]
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    z_axis = np.float32([
        2 * x * z - 2 * y * w,
        2 * y * z + 2 * x * w,
        1 - 2 * x * x - 2 * y * y
    ])
    return z_axis


def colmap_raw_pose_to_RT(image_pose):
    qvec = image_pose[: 4]
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w
        ],
        [
            2 * x * y + 2 * z * w,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * x * w
        ],
        [
            2 * x * z - 2 * y * w,
            2 * y * z + 2 * x * w,
            1 - 2 * x * x - 2 * y * y
        ]
    ])
    # principal_axis.append(R[2, :])
    t = image_pose[4: 7]
    # World-to-Camera pose
    current_pose = np.eye(4)
    current_pose[: 3, : 3] = R
    current_pose[: 3, 3] = t
    return current_pose


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.megasynth_dir, args.precomputed_pairs, args.output_dir)
