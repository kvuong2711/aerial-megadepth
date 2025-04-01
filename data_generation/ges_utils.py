from pymap3d.enu import geodetic2enu
from scipy.spatial.transform import Rotation
import numpy as np
import math
import os
import json
import open3d as o3d
from hloc.utils.read_write_model import Camera, Image, Point3D, read_model, write_model, rotmat2qvec, qvec2rotmat


def rot_ecef2enu(lat, lon):
    lamb = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    sL = np.sin(lamb)
    sP = np.sin(phi)
    cL = np.cos(lamb)
    cP = np.cos(phi)
    rot = np.array([
        [     -sL,       cL,  0],
        [-sP * cL, -sP * sL, cP],
        [ cP * cL,  cP * sL, sP],
    ])
    return rot

def compute_w2c_ecef(rx, ry, rz):
    R = Rotation.from_euler('XYZ', [rx, ry, rz], degrees=True).as_matrix()
    return R


def draw_camera(K, R, t, w, h,
                scale=1.0, color=[0.8, 0.2, 0.8], axis_only=False):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8 * scale)
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = o3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_in_world),
        lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    if axis_only:
        return [axis]

    # return as list in Open3D format
    return [axis, plane, line_set]
    # return [plane, line_set]


def json_to_empty_colmap_model(json_file, ref_sfm_empty, max_num_images=200):
    with open(json_file, "rb") as f:
        raw_tracking_data = json.load(f)

    scene_name = raw_tracking_data["name"]
    w     = raw_tracking_data["width"]
    h     = raw_tracking_data["height"]

    # Take the latitude, longitude, and altitude from the first frame
    lat0, lon0, alt0 = raw_tracking_data['cameraFrames'][0]['coordinate']['latitude'], raw_tracking_data['cameraFrames'][0]['coordinate']['longitude'], raw_tracking_data['cameraFrames'][0]['coordinate']['altitude']

    print('Latitude:', lat0)
    print('Longitude:', lon0)
    print('Altitude:', alt0)

    rot = rot_ecef2enu(lat0, lon0)

    geometries = []
    poses = []

    # create colmap images
    images = {}
    cameras = {}
    cam_id = 1

    for i, frame in enumerate(raw_tracking_data["cameraFrames"]):
        if max_num_images is not None and i > max_num_images:
            break
        x, y, z = geodetic2enu(
            frame['coordinate']['latitude'],
            frame['coordinate']['longitude'],
            frame['coordinate']['altitude'], 
            lat0, lon0, alt0
        )
        dist_scale = 1
        x, y, z = x / dist_scale, y / dist_scale, z / dist_scale
        rx, ry, rz = frame['rotation']['x'], frame['rotation']['y'], frame['rotation']['z']
        R = compute_w2c_ecef(rx, ry, rz)
        c2w = np.block([
            [rot @ R, np.array([x, y, z]).reshape(-1, 1)],
            [np.zeros((1, 3)), 1]
        ])
        
        # Every frame is a camera
        fov_v = frame["fovVertical"]
        theta_v_rad = math.radians(fov_v)
        fl = h / (2 * math.tan(theta_v_rad / 2))
        cx, cy = w / 2, h / 2

        curr_cam = Camera(
            id=cam_id,
            model="SIMPLE_PINHOLE",
            width=w,
            height=h,
            params=np.array([fl, cx, cy])
        )
        cameras[cam_id] = curr_cam
        
        w2c = np.linalg.inv(c2w)
        images[i + 1] = Image(
            id=i + 1,
            qvec=rotmat2qvec(w2c[0:3, 0:3]),
            tvec=w2c[0:3, 3],
            camera_id=cam_id,
            name='{}_{:03d}.jpeg'.format(scene_name, i),
            xys=np.empty([0, 2]),
            point3D_ids=np.empty(0),
        )
        cam_id += 1

        K = np.array([
            [fl, 0, cx],
            [0, fl, cy],
            [0, 0, 1]
        ])

        poses.append(c2w)

        # camera_geom = draw_camera(K=K, R=c2w[0:3, 0:3], t=c2w[0:3, 3], w=w, h=h, scale=10)
        # geometries.extend(camera_geom)

    # camera_geom = draw_camera(K=K, R=np.eye(3), t=c2w[0:3, 3], w=w, h=h, scale=30, axis_only=True)
    # geometries.extend(camera_geom)

    # create COLMAP points3D
    points3D = {}
    print(">>> CAMERAS: ", len(cameras))
    print(">>> IMAGES: ", len(images))
    print(">>> POINTS3D: ", len(points3D))
    os.makedirs(ref_sfm_empty, exist_ok=True)
    # write_model(cameras, images, points3D, ref_sfm_empty, ".txt")
    write_model(cameras, images, points3D, ref_sfm_empty, ".bin")