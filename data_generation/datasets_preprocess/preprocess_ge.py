import os
import glob
import shutil
import argparse
import numpy as np
import cv2


def load_scene_list(scene_list_path):
    loaded = np.load(scene_list_path, allow_pickle=True)
    megadepth_data_dict = loaded['data'].item()
    # scene names
    scene_names = list(megadepth_data_dict.keys())
    scene_names = [scene_name[0] for scene_name in scene_names] # (scene_id, subscene_id)
    return scene_names



def extract_frames_and_copy_json(data_root, downloaded_data_folder, extracted_data_folder, scene_list=None):
    scenes = scene_list
    print(f"Using {len(scenes)} scenes from scene list.")

    for scene_name in scenes:
        scene_mp4_file = os.path.join(downloaded_data_folder, f'{scene_name}.mp4')
        if not os.path.exists(scene_mp4_file):
            print(f'[!] Skipping {scene_name}: .mp4 not found')
            continue

        print(f'>>>> Processing {scene_name}')
        output_scene_folder = os.path.join(extracted_data_folder, scene_name)
        output_footage_folder = os.path.join(output_scene_folder, 'footage')
        os.makedirs(output_footage_folder, exist_ok=True)

        vidcap = cv2.VideoCapture(scene_mp4_file)
        success, image = vidcap.read()
        count = 0

        while success:
            frame_path = os.path.join(output_footage_folder, f'{scene_name}_{count:03d}.jpeg')
            cv2.imwrite(frame_path, image)
            success, image = vidcap.read()
            count += 1

        print(f'Extracted {count} frames to {output_footage_folder}')

        json_file = os.path.join(downloaded_data_folder, f'{scene_name}.json')
        if os.path.exists(json_file):
            shutil.copy(json_file, os.path.join(output_scene_folder, f'{scene_name}.json'))
            print(f'Copied metadata: {json_file}')
        else:
            print(f'Warning: JSON file not found for {scene_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from .mp4 videos and copy corresponding JSON metadata.')
    parser.add_argument('--data_root', type=str, required=True, help='Root path (contains downloaded_data/ and data/ folders)')
    parser.add_argument('--scene_list', type=str, required=True, help='Path to .npz file listing scene names to process')
    args = parser.parse_args()

    scene_list = load_scene_list(args.scene_list) if args.scene_list else None
    print(f"Using scene list: {scene_list}")

    # Extract frames and copy JSON metadata
    downloaded_data_folder = os.path.join(args.data_root, 'downloaded_data')
    assert os.path.exists(downloaded_data_folder), f"Downloaded data folder not found: {downloaded_data_folder}"

    extracted_data_folder = os.path.join(args.data_root, 'data')
    os.makedirs(extracted_data_folder, exist_ok=True)

    extract_frames_and_copy_json(data_root=args.data_root, 
                                 downloaded_data_folder=downloaded_data_folder, 
                                 extracted_data_folder=extracted_data_folder, 
                                 scene_list=scene_list)
