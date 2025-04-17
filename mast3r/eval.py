import argparse
import mast3r.utils.path_to_dust3r
from mast3r.model import AsymmetricMASt3R
import torch
import numpy as np
import os
import os.path as osp
import json
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from eval_utils import camera_to_rel_deg
from colmap_utils import read_model, qvec2rotmat, get_camera_matrix


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--eval_data_dir', type=str, required=True)
    return parser


def post_process(output, device, ga_mode=GlobalAlignerMode.PairViewer):
    # # at this stage, you have the raw dust3r predictions
    # view1, pred1 = output['view1'], output['pred1']
    # view2, pred2 = output['view2'], output['pred2']
    
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    if ga_mode == GlobalAlignerMode.PointCloudOptimizer:
        # With GA
        schedule = 'cosine'
        lr = 0.01
        niter = 300
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    elif ga_mode == GlobalAlignerMode.PairViewer:
        # No GA
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=True)
    else:
        raise NotImplementedError(f'Unknown mode {ga_mode}')

    # retrieve useful values from scene:
    poses = scene.get_im_poses()

    # if it's a parameter, then take the data
    if isinstance(poses, torch.nn.Parameter):
        poses = poses.data

    return poses.detach(), scene



def load_colmap_data(colmap_data_folder):
    # print('Loading COLMAP data...')
    input_format = '.bin'
    cameras, images, points3D = read_model(colmap_data_folder, ext=input_format)
    # print(f'num_cameras: {len(cameras)}')
    # print(f'num_images: {len(images)}')
    # print(f'num_points3D: {len(points3D)}')

    colmap_pose_dict = {}

    # Loop through COLMAP images
    for img_id, img_info in images.items():
        img_name = img_info.name

        C_R_G, C_t_G = qvec2rotmat(img_info.qvec), img_info.tvec

        # invert
        G_t_C = -C_R_G.T @ C_t_G
        G_R_C = C_R_G.T

        cam_info = cameras[img_info.camera_id]
        cam_params = cam_info.params
        K, _ = get_camera_matrix(camera_params=cam_params, camera_model=cam_info.model)

        colmap_pose_dict[img_name] = (K, G_R_C, G_t_C)

    return colmap_pose_dict, points3D



def setup_test_data(root_dir, split_name):
    all_test_data = {}

    #### siteACC0002-mall ####
    data_folder = osp.join(root_dir, 'siteACC0002-mall')
    input_image_folder = osp.join(data_folder, 'images')
    colmap_folder = osp.join(data_folder, 'model')

    pairs_file_txt = os.path.join(data_folder, f'{split_name}_covis_pairs.txt')
    pairs = []
    with open(pairs_file_txt, 'r') as f:
        for line in f:
            img1, img2 = line.strip().split()
            pairs.append((img1, img2))
    print('Loaded {} pairs for data_folder: {}'.format(len(pairs), data_folder))
    wriva_pose_dict, _ = load_colmap_data(colmap_folder)

    all_test_data['siteACC0002-mall'] = {'input_image_folder': input_image_folder, 'wriva_pose_dict': wriva_pose_dict, 'pairs': pairs}
    #### siteACC0002-mall ####

    #### siteACC0003-finearts ####
    data_folder = osp.join(root_dir, 'siteACC0003-finearts')
    input_image_folder = osp.join(data_folder, 'images')
    colmap_folder = osp.join(data_folder, 'model')

    pairs_file_txt = os.path.join(data_folder, f'{split_name}_covis_pairs.txt')
    pairs = []
    with open(pairs_file_txt, 'r') as f:
        for line in f:
            img1, img2 = line.strip().split()
            pairs.append((img1, img2))
    print('Loaded {} pairs for data_folder: {}'.format(len(pairs), data_folder))

    # Load the WRIVA data
    wriva_pose_dict, _ = load_colmap_data(colmap_folder)

    all_test_data['siteACC0003-finearts'] = {'input_image_folder': input_image_folder, 'wriva_pose_dict': wriva_pose_dict, 'pairs': pairs}
    #### siteACC0003-finearts ####

    #### siteM07-door ####
    data_folder = osp.join(root_dir, 'siteM07-door')
    input_image_folder = osp.join(data_folder, 'images')
    colmap_folder = osp.join(data_folder, 'model')

    pairs_file_txt = os.path.join(data_folder, f'{split_name}_covis_pairs.txt')
    pairs = []
    with open(pairs_file_txt, 'r') as f:
        for line in f:
            img1, img2 = line.strip().split()
            pairs.append((img1, img2))
    print('Loaded {} pairs for data_folder: {}'.format(len(pairs), data_folder))
    wriva_pose_dict, _ = load_colmap_data(colmap_folder)

    all_test_data['siteM07-door'] = {'input_image_folder': input_image_folder, 'wriva_pose_dict': wriva_pose_dict, 'pairs': pairs}
    #### siteM07-door ####

    #### t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10 ####
    data_folder = osp.join(root_dir, 't04_v13_s00_r01_VaryingAltitudes_WACV_test_A10')
    input_image_folder = osp.join(data_folder, 'images')
    colmap_folder = osp.join(data_folder, 'model')

    pairs_file_txt = os.path.join(data_folder, f'{split_name}_covis_pairs.txt')
    pairs = []
    with open(pairs_file_txt, 'r') as f:
        for line in f:
            img1, img2 = line.strip().split()
            pairs.append((img1, img2))
    print('Loaded {} pairs for data_folder: {}'.format(len(pairs), data_folder))
    wriva_pose_dict, _ = load_colmap_data(colmap_folder)

    all_test_data['t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10'] = {'input_image_folder': input_image_folder, 'wriva_pose_dict': wriva_pose_dict, 'pairs': pairs}
    #### t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10 ####

    return all_test_data



def test_fn(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load the data
    test_split = 'ground_aerial'
    print('>>> Loading data with split: {}'.format(test_split))
    root_dir = args.eval_data_dir

    all_test_data = setup_test_data(root_dir, test_split)

    # model
    model = AsymmetricCroCo3DStereo.from_pretrained(args.weights).to(device)
    model.eval()    

    error_dict = {"rError": [], "tError": []}

    for data_key in all_test_data:
        input_image_folder, wriva_pose_dict, pairs = all_test_data[data_key]['input_image_folder'], all_test_data[data_key]['wriva_pose_dict'], all_test_data[data_key]['pairs']

        print('>>>> Testing on input_image_folder {}, with {} pairs'.format(input_image_folder, len(pairs)))

        # Loop through the pairs
        for pair_idx in tqdm(range(len(pairs))):
            img1, img2 = pairs[pair_idx]
            img1_fullpath = os.path.join(input_image_folder, img1)
            img2_fullpath = os.path.join(input_image_folder, img2)

            if not osp.exists(img1_fullpath) or not osp.exists(img2_fullpath):
                print(f'Image not found: {img1_fullpath} or {img2_fullpath}')
                continue

            # Load the images and run inference
            images = load_images([img1_fullpath, img2_fullpath], size=512)
            input_pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(input_pairs, model, device, batch_size=1)
            pred_poses, scene = post_process(output, device, ga_mode=GlobalAlignerMode.PairViewer)  # GlobalAlignerMode.PointCloudOptimizer
            # pred_poses = post_process(output, device, ga_mode=GlobalAlignerMode.PointCloudOptimizer)  # GlobalAlignerMode.PointCloudOptimizer

            # Load the ground truth poses
            G_R_C1, G_t_C1 = wriva_pose_dict[img1][1], wriva_pose_dict[img1][2]
            G_R_C2, G_t_C2 = wriva_pose_dict[img2][1], wriva_pose_dict[img2][2]
            G_T_C1 = np.eye(4)
            G_T_C1[:3, :4] = np.hstack((G_R_C1, G_t_C1.reshape(3, 1)))
            G_T_C2 = np.eye(4)
            G_T_C2[:3, :4] = np.hstack((G_R_C2, G_t_C2.reshape(3, 1)))
            gt_poses = np.array([G_T_C1, G_T_C2], dtype=np.float32)
            gt_poses = torch.from_numpy(gt_poses).to(device)

            # Compute the error
            rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_poses, gt_poses, device, batch_size=1)
            print(f'{img1} -> {img2}')
            print(f"    --  Mean Rot   Error (Deg) for this scene: {rel_rangle_deg.mean():10.2f}")
            print(f"    --  Mean Trans Error (Deg) for this scene: {rel_tangle_deg.mean():10.2f}")

            rel_rangle_deg = rel_rangle_deg.cpu().numpy()
            assert rel_rangle_deg.shape[0] == 1
            rel_tangle_deg = rel_tangle_deg.cpu().numpy()[0]

            error_dict["rError"].append(float(rel_rangle_deg))
            error_dict["tError"].append(float(rel_tangle_deg))

    # Compute RRA and RTA @ (2, 5, 10, 15, 30) degrees
    rError = np.array(error_dict['rError'])
    tError = np.array(error_dict['tError'])

    Racc_5 = np.mean(rError < 5) * 100
    Racc_10 = np.mean(rError < 10) * 100
    Racc_15 = np.mean(rError < 15) * 100
    Racc_30 = np.mean(rError < 30) * 100

    Tacc_5 = np.mean(tError < 5) * 100
    Tacc_10 = np.mean(tError < 10) * 100
    Tacc_15 = np.mean(tError < 15) * 100
    Tacc_30 = np.mean(tError < 30) * 100

    print(f"RRA @ 5 deg: {Racc_5:10.2f}")
    print(f"RRA @ 10 deg: {Racc_10:10.2f}")
    print(f"RRA @ 15 deg: {Racc_15:10.2f}")
    print(f"RRA @ 30 deg: {Racc_30:10.2f}")
    print('---------------------------------')
    print(f"RTA @ 5 deg: {Tacc_5:10.2f}")
    print(f"RTA @ 10 deg: {Tacc_10:10.2f}")
    print(f"RTA @ 15 deg: {Tacc_15:10.2f}")
    print(f"RTA @ 30 deg: {Tacc_30:10.2f}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    test_fn(args)