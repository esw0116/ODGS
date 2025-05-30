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

import torch
from scene_perspective import Scene
from scene_perspective.gaussian_model import GaussianModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer.pinhole_renderer import pinhole_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import numpy as np
import matplotlib


cmapper = matplotlib.cm.get_cmap('jet_r')

def depth_colorize_with_mask(depthlist, background=(0,0,0), dmindmax=None):
    """ depth: (H,W) - [0 ~ 1] / mask: (H,W) - [0 or 1]  -> colorized depth (H,W,3) [0 ~ 1] """
    single_batch = True if len(depthlist.shape)==2 else False
        
    if single_batch:
        depthlist = depthlist[None]
    
    batch, vx, vy = np.where(depthlist!=0)
    if dmindmax is None:
        valid_depth = depthlist[batch, vx, vy]
        dmin, dmax = valid_depth.min(), valid_depth.max()
    else:
        dmin, dmax = dmindmax
    
    norm_dth = np.ones_like(depthlist)*dmax # [B, H, W]
    norm_dth[batch, vx, vy] = (depthlist[batch, vx, vy]-dmin)/(dmax-dmin)
    
    final_depth = np.ones(depthlist.shape + (3,)) * np.array(background).reshape(1,1,1,3) # [B, H, W, 3]
    final_depth[batch, vx, vy] = cmapper(norm_dth)[batch,vx,vy,:3]

    return final_depth[0] if single_batch else final_depth


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, rand): 
    rotate_mode = ['F', 'R', 'L', 'B', 'U', 'D']
    
    # raise("check - save original pinhole")
    if rand:
        render_path = os.path.join(model_path, name, "ours_randpin_{}".format(iteration), "renders")
    else:
        render_path = os.path.join(model_path, name, "ours_pinhole_{}".format(iteration), "renders")
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    # makedirs(depth_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ncols=110)):
        results = pinhole_render(view, gaussians, pipeline, background, None)
        rendering = results["render"]
        # gt = view.original_image[0:3, :, :]
        
        image = torch.clamp(rendering, 0.0, 1.0)
        image_name = view.image_name
        # direction = image_name.split('_')[-1]
        # cube_dict[direction] = frac_image
        
        # if idx % 6 < 5:
        #     continue
        # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        # gt_image = torch.clamp(view.original_erp_image.to("cuda"), 0.0, 1.0)
        # image = equilib.cube2equi(cube_dict, cube_format='dict', height=gt_image.shape[-2], width=gt_image.shape[-1])
        
        
        # torchvision.utils.save_image(rendering, os.path.join(render_path, f"{view.erp_image_name}.png"))
        torchvision.utils.save_image(image, os.path.join(render_path, f"{image_name}.png"))

    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    #     for rot_sym in rotate_mode:
    #         results = pinhole_render(view, gaussians, pipeline, background, rot_sym)
    #         rendering = results["render"]
    #         # gt = view.original_image[0:3, :, :]
            
    #         torchvision.utils.save_image(rendering, os.path.join(render_path, '{}'.format(view.image_name) + ".png"))
    #         # torchvision.utils.save_image(gt, os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, preset : bool, scale_max: float, rand: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, scale_max=scale_max)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, rand=rand)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if preset:
            render_set(dataset.model_path, "preset", scene.loaded_iter, scene.getSample1Cameras(), gaussians, pipeline, background, rand)
        else:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, rand)
            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, rand)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--preset", action="store_true")
    parser.add_argument("--rand", action="store_true")
    parser.add_argument("--scale_max", default=100, type=float)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.preset, args.scale_max, args.rand)
    
    """
    ## command: 
    CUDA_VISIBLE_DEVICES=1 python render.py --eval -m ./output/bicycle360 --iteration 7000
    """