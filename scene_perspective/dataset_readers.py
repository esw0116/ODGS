#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    erp_image_name: str

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readCamerasFromOpenMVG(path, extrinsicsfile, cam_dict, white_background):
    cam_infos = []

    with open(os.path.join(path, extrinsicsfile)) as json_file:
        contents = json.load(json_file)
        fovx = 3.13768641

        frames = contents["extrinsics"]
        for idx, frame in enumerate(frames):
            cam_key = frame["key"]
            cam_name = os.path.join(path, 'images', cam_dict[cam_key])
            
            R = np.array(frame["value"]["rotation"]).T
            T = -np.array(frame["value"]["rotation"]) @ np.array(frame["value"]["center"])

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], erp_image_name=image_name))
            
    return cam_infos


def readOpenMVGInfo(path, white_background, eval, use_dense):
    print("Reading Transforms from OpenMVG")

    my_views = os.path.join(path, "data_views.json")
    camfile_dict = {}
    with open(my_views) as views:
        json_views = json.load(views)
        camview_list = json_views["views"]
        for camview in camview_list:
            camfile_dict[camview["key"]] = camview["value"]["ptr_wrapper"]["data"]["filename"]

    cam_infos_unsorted = readCamerasFromOpenMVG(path, "data_extrinsics.json", camfile_dict, white_background)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    try:
        train_file = os.path.join(path, 'train.txt')
        test_file = os.path.join(path, 'test.txt')
        with open(train_file, 'r') as f:
            train_name_list = f.read().splitlines() 
        with open(test_file, 'r') as f:
            test_name_list = f.read().splitlines() 
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_name_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_name_list]
        
    except:
        raise AssertionError("Please Specify train test split")
        
    print(f"# of Train: {len(train_cam_infos)}, \t# of Test: {len(test_cam_infos)}")

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if use_dense:
        if os.path.exists(os.path.join(path, "scene_dense_SGM.ply")):
            print("Dense pcd (SGM) is initialized")
            ply_path = os.path.join(path, "scene_dense_SGM.ply")
        elif os.path.exists(os.path.join(path, "scene_dense.ply")):
            print("Dense pcd is initialized")
            ply_path = os.path.join(path, "scene_dense.ply")
    else:
        if os.path.exists(os.path.join(path, "pcd.ply")):
            print("Points without camera position (Green points) are initialized")
            ply_path = os.path.join(path, "pcd.ply")
        else:
            ply_path = os.path.join(path, "colorized.ply")

    if not os.path.exists(ply_path):
        raise FileNotFoundError('No initial pcd file found!')
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromCubemap(path, extrinsicsfile, cam_dict, white_background):
    cam_infos = []

    with open(os.path.join(path, extrinsicsfile)) as json_file:
        contents = json.load(json_file)
        fovx = 3.1415926 / 2

        frames = contents["extrinsics"]
        for idx, frame in enumerate(frames):
            cam_key = frame["key"]
            cam_name = os.path.join(path, 'cubemap', cam_dict[cam_key])
            
            R = np.array(frame["value"]["rotation"]).T
            T = -np.array(frame["value"]["rotation"]) @ np.array(frame["value"]["center"])

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            
            erp_name = cam_dict[cam_key][:-6] + cam_dict[cam_key][-4:]
            erp_cam_name = os.path.join(path, 'images', erp_name)
            erp_image_name = Path(erp_cam_name).stem
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], erp_image_name=erp_image_name))
    return cam_infos


def readCubeMapInfo(path, white_background, eval, use_dense, rand=False):
    print("Reading Transforms from Cubemaps")

    if rand:
        my_views = os.path.join(path, "cube_randrotate_data_views.json")
    else:
        my_views = os.path.join(path, "cube_data_views.json")
    camfile_dict = {}
    with open(my_views) as views:
        json_views = json.load(views)
        camview_list = json_views["views"]
        for camview in camview_list:
            camfile_dict[camview["key"]] = camview["value"]["ptr_wrapper"]["data"]["filename"]
    
    if rand:
        cam_infos_unsorted = readCamerasFromCubemap(path, "cube_randrotate_data_extrinsics.json", camfile_dict, white_background)

    else:
        cam_infos_unsorted = readCamerasFromCubemap(path, "cube_data_extrinsics.json", camfile_dict, white_background)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    try:
        train_file = os.path.join(path, 'train_reduced.txt')
        with open(train_file, 'r') as f:
            train_name_list = f.read().splitlines() 
    except:
        train_file = os.path.join(path, 'train.txt')
        with open(train_file, 'r') as f:
            train_name_list = f.read().splitlines() 
    
    test_file = os.path.join(path, 'test.txt')
    with open(test_file, 'r') as f:
        test_name_list = f.read().splitlines()
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.erp_image_name in train_name_list]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.erp_image_name in test_name_list]
    
    print(f"# of Train: {len(train_cam_infos)}, \t# of Test: {len(test_cam_infos)}")

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if use_dense:
        if os.path.exists(os.path.join(path, "scene_dense_SGM.ply")):
            print("Dense pcd (SGM) is initialized")
            ply_path = os.path.join(path, "scene_dense_SGM.ply")
        elif os.path.exists(os.path.join(path, "scene_dense.ply")):
            print("Dense pcd is initialized")
            ply_path = os.path.join(path, "scene_dense.ply")
    else:
        if os.path.exists(os.path.join(path, "pcd.ply")):
            print("Points without camera position (Green points) are initialized")
            ply_path = os.path.join(path, "pcd.ply")
        else:
            ply_path = os.path.join(path, "colorized.ply")

    if not os.path.exists(ply_path):
        raise FileNotFoundError('No initial pcd file found!')
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "OpenMVG" : readOpenMVGInfo,
    "CubeMap" : readCubeMapInfo,
}