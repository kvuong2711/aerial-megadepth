import argparse
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.fast_nn import extract_correspondences_nonsym
from hloc_viz import plot_matches, plot_images
import matplotlib.pyplot as plt

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    weights_path = args.weights

    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)

    # Potential input images to test  
    image_list = [
        '../assets/t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10/00572.jpg',
        '../assets/t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10/00563.jpg'
    ]  

    # image_list = [
    #     '../assets/t03_v07_s00_r01_ReconstructedArea_WACV_test_A09/00412.jpg',
    #     '../assets/t03_v07_s00_r01_ReconstructedArea_WACV_test_A09/00399.jpg'
    # ]

    # image_list = [
    #     '../assets/t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10/00616.jpg',
    #     '../assets/t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10/00563.jpg'
    # ]
    
    # image_list = [
    #         '../assets/siteACC0002-mall/siteACC0002-camA005-2023-10-17-15-29-00-000283.jpg',
    #         '../assets/siteACC0002-mall/siteACC0002-camA010-2023-12-21-19-29-06-000486.jpg'
    # ]

    # image_list = [
    #         '../assets/siteACC0003-finearts/siteACC0003-camA005-2023-10-16-13-32-00-000181.jpg',
    #         '../assets/siteACC0003-finearts/siteACC0003-camA010-2023-12-21-19-43-25-000309.jpg', 
    # ]

    # image_list = [
    #         '../assets/siteACC0003-finearts/siteACC0003-camA005-2023-10-19-11-38-16-000205.jpg',,
    #         '../assets/siteACC0003-finearts/siteACC0003-camA010-2023-12-21-19-43-25-000704.jpg',
    # ]

    # image_list = [   
    #         '../assets/siteACC0153-austin-rec-center/siteACC0153-camA006-2024-04-23-09-35-14-000234.jpg',
    #         '../assets/siteACC0153-austin-rec-center/siteACC0153-camA011-2024-04-24-12-07-02-000258.jpg'
    # ]

    # image_list = [
    #         '../assets/t04_v07_s02_r02_VaryingAltitudes_M07_building_1_door/image_000020.jpg',
    #         '../assets/t04_v07_s02_r02_VaryingAltitudes_M07_building_1_door/image_000003.jpg'
    # ]

    # image_list = [
    #         '../assets/WGLBS/maidan/03.png',
    #         '../assets/WGLBS/maidan/01.png',
    # ]

    images = load_images(image_list, size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    ######### Visualize the 2D matches extracted from MASt3R's local feature maps
    subsample = 4  # subsample rate, the lower the denser the matches
    pixel_tol = 3
    match_conf = 0.6

    corres = extract_correspondences_nonsym(pred1['desc'][0], pred2['desc'][0], 
                                            pred1['desc_conf'][0], pred2['desc_conf'][0],
                                            device=device, subsample=subsample, pixel_tol=pixel_tol)
    conf = corres[2]
    mask = conf >= match_conf
    matches_im0 = corres[0][mask].cpu().numpy()
    matches_im1 = corres[1][mask].cpu().numpy()

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    plot_images(viz_imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True)
    plot_matches(matches_im0, matches_im1, color='lime', lw=0.3, ps=2, a=0.9) # , ps=4, indices=(0, 1), a=0.5)

    plt.show()