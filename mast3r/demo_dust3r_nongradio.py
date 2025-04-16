import argparse
import mast3r.utils.path_to_dust3r
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
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
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # load the model
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)

    # Potential input images to test
    ###### Pairwise #####
    image_list = [
            '../assets/siteACC0003-finearts/siteACC0003-camA005-2023-10-16-13-32-00-000181.jpg',
            '../assets/siteACC0003-finearts/siteACC0003-camA010-2023-12-21-19-43-25-000309.jpg', 
    ]

    # image_list = [
    #         '../assets/siteACC0002-mall/siteACC0002-camA005-2023-10-17-15-29-00-000283.jpg',
    #         '../assets/siteACC0002-mall/siteACC0002-camA010-2023-12-21-19-29-06-000486.jpg'
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

    # ###### Multi-view (optimized using GlobalAligner) ######
    # image_list = [
    #         '../assets/siteACC0003-camA010-2023-12-21-19-43-25-000307.jpg',
    #         '../assets/siteACC0003-camA005-2023-10-16-14-03-00-000176.jpg', 
    #         '../assets/siteACC0003-camA005-2023-10-16-13-32-00-000201.jpg',
    #         '../assets/siteACC0003-camA005-2023-10-16-12-33-00-000267.jpg',
    # ]
    
    # image_list = [
    #     '../assets/siteACC0002-mall/merged_data/siteACC0002-camA010-2023-12-21-19-29-06-000932.jpg', 
    #     '../assets/siteACC0002-mall/merged_data/siteACC0002-camA005-2023-10-17-15-29-00-000293.jpg',
    #     '../assets/siteACC0002-mall/merged_data/siteACC0002-camA005-2023-10-17-15-50-00-000171.jpg',
    #     '../assets/siteACC0002-mall/merged_data/siteACC0002-camA005-2023-10-19-10-48-18-000001.jpg'
    # ]

    images = load_images(image_list, size=512)

    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
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
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()        
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # scene.min_conf_thr = 1.5

    # visualize reconstruction
    scene.show()
