import importlib
TSDFPlanner = importlib.import_module('explore-eqa.src.tsdf').TSDFPlanner
from RoboInfoGather.program_utils import *
from RoboInfoGather.pomdp import *
from RoboInfoGather.MCTS_planner import *
from RoboInfoGather.map_utils import *
from RoboInfoGather.observation_utils import *
from RoboInfoGather.info_gather_runner import *

import logging

import csv
import pickle
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import imageio as iio
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from collections import OrderedDict

def test_frontier(
        cfg,
        env,
        camera_data,
        scene_data,
        task_info,
        cum_sim_score,
        cnt_data,
        position_data,
        vlm_models,
        question_ind
    ):

    ################################
    # Set Up Planner from Question #
    ################################

    # Initialize TSDF
    tsdf_planner = TSDFPlanner(
        vol_bnds=scene_data['tsdf_bnds'],
        voxel_size=cfg.tsdf_grid_size,
        floor_height_offset=0,
        pts_init=position_data['pts'],
        init_clearance=0.5,
    )

    # Run steps
    cnt_step = 0
    num_black_in_a_row = 0
    responses = []
    result = {}
    print("Starting While Loop")
    pts = position_data['pts']

    angle = position_data['angle']
    pitch = position_data['pitch']
    roll = position_data['roll']
    while cnt_step < scene_data['num_step']:
        logging.info(f"\n== step: {cnt_step}")
        # Update environment
        pts, angle, cam_pose, cam_pose_tsdf, camera_pos, obs = env_update(
                cnt_step,
                pts,
                pitch,
                roll,
                angle,
                env
            )
        step_name = f"step_{cnt_step}"
        result[step_name] = {"pts": pts, "angle": angle}

        plt.imshow(obs['rgb'])
        plt.show()

        # TSDF fusion
        tsdf_planner.integrate(
            color_im=obs['rgb'],
            depth_im=obs['depth'],
            cam_intr=camera_data['cam_intr'],
            cam_pose=cam_pose_tsdf,
            obs_weight=1.0,
            margin_h=int(cfg.margin_h_ratio * camera_data['img_data']['h']),
            margin_w=int(cfg.margin_w_ratio * camera_data['img_data']['w']),
        )

        ########################
        # Determine next point #
        ########################
        print("Starting VLM check")
        pts, angle, fig = get_next_point(
                obs['rgb'],
                task_info['question'],
                pts,
                cam_pose_tsdf,
                camera_pos,
                cam_pose,
                tsdf_planner,
                camera_data['img_data']['w'],
                camera_data['img_data']['h'],
                camera_data['cam_intr'],
                cfg,
                task_info['episode_data_dir'],
                cnt_step,
                vlm_models['vlm'],
                obs['depth'],
                pts,
                angle,
                None,
                scene_data
            )

        cnt_step += 1
        print("Done Iteration: ", cnt_step)

    return None
