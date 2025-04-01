import argparse
import multiprocessing
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap

from . import logger
from .triangulation import (
    OutputCapture,
    estimation_and_geometric_verification,
    import_features,
    import_matches,
    parse_option_args,
)
from .utils.database import COLMAPDatabase
from PIL import Image
import numpy as np
import subprocess
from .utils.read_write_model import read_cameras_binary, read_images_binary
import pprint


class CalledProcessError(subprocess.CalledProcessError):
    def __str__(self):
        message = "Command '%s' returned non-zero exit status %d." % (
                ' '.join(self.cmd), self.returncode)
        if self.output is not None:
            message += ' Last outputs:\n%s' % (
                '\n'.join(self.output.decode('utf-8').split('\n')[-10:]))
        return message

# TODO: consider creating a Colmap object that holds the path and verbose flag
def run_command(cmd, verbose=False):
    stdout = None if verbose else subprocess.PIPE
    ret = subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=stdout)
    if not ret.returncode == 0:
        raise CalledProcessError(
                returncode=ret.returncode, cmd=cmd, output=ret.stdout)
    

def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    logger.info("Creating an empty database...")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(
    colmap_path, 
    sfm_dir,
    image_dir: Path,
    database_path: Path,
    original_image_ids: List[int],
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    logger.info("Importing images into the database...")
    # if options is None:
    #     options = {}
    # images = list(image_dir.iterdir())
    # if len(images) == 0:
    #     raise IOError(f"No images found in {image_dir}.")
    # with pycolmap.ostream():
    #     pycolmap.import_images(
    #         database_path,
    #         image_dir,
    #         camera_mode,
    #         image_list=image_list or [],
    #         options=options,
    #     )

    db = COLMAPDatabase.connect(database_path)
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")
    
    cam_id_start = 1000
    image_id_start = 1000

    for i, name in enumerate(sorted(images)):
        if name.is_file() and name.suffix in ['.png', '.jpg', '.jpeg']:
            width, height = Image.open(name).size
            # Load npz as well
            npz_path = str(name)[:-4] + '.npz'
            intrinsics = np.load(npz_path)['intrinsics']
            params = (intrinsics[0, 0], intrinsics[0, 2], intrinsics[1, 2])
            cam_tuple = (0, width, height, params)
            db.add_camera(*cam_tuple, camera_id=cam_id_start + i, prior_focal_length=True)

            # Add image
            base_name = name.name
            db.add_image(str(base_name), camera_id=cam_id_start + i, image_id=image_id_start + i)

    db.commit()
    db.close()
    


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def run_reconstruction(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction...")
    if options is None:
        options = {}
    options = {"num_threads": min(multiprocessing.cpu_count(), 16), **options}
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstructions = pycolmap.incremental_mapping(
                database_path, image_dir, models_path, options=options
            )

    if len(reconstructions) == 0:
        logger.error("Could not reconstruct any model!")
        return None
    logger.info(f"Reconstructed {len(reconstructions)} model(s).")

    largest_index = None
    largest_num_images = 0
    for index, rec in reconstructions.items():
        num_images = rec.num_reg_images()
        if num_images > largest_num_images:
            largest_index = index
            largest_num_images = num_images
    assert largest_index is not None
    logger.info(
        f"Largest model is #{largest_index} " f"with {largest_num_images} images."
    )

    for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
        if (sfm_dir / filename).exists():
            (sfm_dir / filename).unlink()
        shutil.move(str(models_path / str(largest_index) / filename), str(sfm_dir))
    return reconstructions[largest_index]


def copy_images_to_one_folder(db_image_dir, query_image_dir, output_image_dir):
    logger.info(f'Copying all query and database images into {output_image_dir}.')
    output_image_dir.mkdir(parents=True, exist_ok=True)
    for db_img_path in db_image_dir.iterdir():
        if db_img_path.is_file() and db_img_path.suffix in ['.png', '.jpg', '.jpeg']:
            if not (output_image_dir / db_img_path.name).exists():
                shutil.copy(db_img_path, output_image_dir / db_img_path.name)

    for query_img_path in query_image_dir.iterdir():
        if query_img_path.is_file() and query_img_path.suffix in ['.png', '.jpg', '.jpeg']:
            if not (output_image_dir / query_img_path.name).exists():
                shutil.copy(query_img_path, output_image_dir / query_img_path.name)


def resume_reconstruction(colmap_path, sfm_dir, database_path, sparse_input_model_path, image_dir,
                          min_num_matches=None, verbose=False):
    assert sparse_input_model_path.exists()

    latest_model_path = sfm_dir / 'localized_model_mapper'
    latest_model_path.mkdir(exist_ok=True, parents=True)

    cmd = [
        str(colmap_path), 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--input_path', str(sparse_input_model_path),
        '--output_path', str(latest_model_path),
        '--Mapper.num_threads', str(min(multiprocessing.cpu_count(), 8)),
        '--Mapper.ba_refine_focal_length', 'False',
        '--Mapper.ba_refine_principal_point', 'False',
        '--Mapper.ba_refine_extra_params', 'False',
        '--Mapper.fix_existing_images', 'True',
    ]

    # TODO: bugfix Mapper.fix_existing_images

    if min_num_matches:
        cmd += ['--Mapper.min_num_matches', str(min_num_matches)]
    logger.info('Running the image_registrator with command:\n%s', ' '.join(cmd))
    
    run_command(cmd, verbose)

    # Largest (latest) model analyzer
    largest_model = latest_model_path
    largest_model_num_cams = len(read_cameras_binary(str(largest_model / 'cameras.bin')))
    largest_model_num_images = len(read_images_binary(str(largest_model / 'images.bin')))
    logger.info(f'Largest model is #{largest_model.name} '
                 f'with {largest_model_num_cams} cameras, '
                 f'{largest_model_num_images} images.')

    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer',
         '--path', str(largest_model)])
    stats_raw = stats_raw.decode().split("\n")
    stats = dict()
    for stat in stats_raw:
        if stat.startswith("Registered images"):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith("Points"):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])

    ## Move the model files outside
    # for filename in ['images.bin', 'cameras.bin', 'points3D.bin']:
    #     shutil.move(str(largest_model / filename), str(sfm_dir / filename))

    return stats


def main(
    sfm_dir: Path,
    db_image_dir: Path,
    query_image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    verbose: bool = False,
    skip_geometric_verification: bool = False,
    min_match_score: Optional[float] = None,
    image_list: Optional[List[str]] = None,
    image_options: Optional[Dict[str, Any]] = None,
    mapper_options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    
    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    # Initialize the new database
    database = sfm_dir / 'database.db'
    shutil.copyfile(database, sfm_dir / 'latest_database.db')
    database = sfm_dir / 'latest_database.db'
    print(f"Database path: {database}")

    # a few prelim file existence checks
    assert pairs.exists(), pairs
    assert sfm_dir.exists()
    assert database.exists()

    # get original map image_ids
    original_image_ids_dict = get_image_ids(database)
    original_image_ids_list = list(original_image_ids_dict.values())

    copy_images_to_one_folder(db_image_dir, query_image_dir, sfm_dir / 'images')
    logger.info(f'Images (both query and db) are located at {sfm_dir / "images"}')

    colmap_path = 'colmap'
    import_images(colmap_path, sfm_dir, query_image_dir, database, original_image_ids_list, camera_mode, image_list, image_options)

    image_ids = get_image_ids(database)
    new_image_ids_dict = {k: v for k, v in image_ids.items() if k not in original_image_ids_dict}
    logger.info(f'New images dict: {new_image_ids_dict}')

    import_features(new_image_ids_dict, database, features)

    import_matches(
        image_ids,
        database,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )

    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs, verbose)

    image_dir = sfm_dir / 'images'
    sparse_input_model_path = sfm_dir

    stats = resume_reconstruction(
        colmap_path, sfm_dir, database, sparse_input_model_path, image_dir, None, verbose)
    
    if stats is not None:
        stats['num_input_images'] = len(image_ids)
        logger.info('Reconstruction statistics:\n%s', pprint.pformat(stats))