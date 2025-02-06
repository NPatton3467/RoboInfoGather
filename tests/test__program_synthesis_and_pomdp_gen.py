import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np

np.set_printoptions(precision=3)
import csv
import pickle
import logging
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner

# RoboInfoGather Imports
from groundingdino.util.inference import load_model
from RoboInfoGather.program_utils import *
from RoboInfoGather.pomdp import *
from RoboInfoGather.pomdp_exec import *
from RoboInfoGather.MCTS_planner import *
from RoboInfoGather.map_utils import *

def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)


    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    with open(cfg.init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }

    for question_ind in range(10):

        # Extract question
        question_data = questions_data[question_ind]
        scene = question_data["scene"]
        floor = question_data["floor"]
        scene_floor = scene + "_" + floor
        question = question_data["question"]
        # This was first but doesn't work choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
        choices = question_data["choices"].split(",")
        answer = question_data["answer"]
        text_answer = choices[0]
        if answer == "B":
            text_answer = choices[1]
        elif answer == "C":
            text_answer = choices[2]
        elif answer == "D":
            text_answer = choices[3]
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]

        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        scene_mesh_dir = os.path.join(
            cfg.scene_data_path, scene, scene[6:] + ".basis" + ".glb"
        )
        navmesh_file = os.path.join(
            cfg.scene_data_path, scene, scene[6:] + ".basis" + ".navmesh"
        )
        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": cfg.hfov,
        }
        sim_cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        pathfinder.load_nav_mesh(navmesh_file)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        pts = init_pts
        angle = init_angle

        # Floor - use pts height as floor height
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
        
        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
        )

        ################################
        # Set Up Planner from Question #
        ################################
        # TODO: Planner Setup
        
        # Load the config
        config_filename = os.path.join(f"/robodata/user_data/npatt/explore-eqa/RoboInfoGather/info_gather.yaml")
        RIG_config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        
        # Generate program
        attempts = 0
        prog = None
        while attempts < 30:
            attempts += 1
            try:
                prog = gen_prog_from_nl(question)
                break
            except Exception as e:
                print(f"Could not generate program, attempt {attempts}")

        print(f"\nQuestion:\n{question}\nProgram: ")
        print(prog.pretty_str())
        resolution = cfg.tsdf_grid_size
        size = int(max(abs(tsdf_bnds[0][0] - tsdf_bnds[0][1]), abs(tsdf_bnds[1][0] - tsdf_bnds[1][1])) / resolution)

        #This format: np.array([-way_point.loc.y, pts[1], way_point.loc.x])
        pos = pos_habitat_to_normal(pts)
        if type(prog) is Prog:
            pomdp = gen_pomdp_from_query(
                    query=prog.expressions[0],
                    pos=pos,
                    yaw=angle,
                    trav_map_og_size=size,
                    trav_map_og_res=resolution,
                    configs=RIG_config
                )
        else:
            pomdp = gen_pomdp_from_query(
                    query=prog,
                    pos=pos,
                    yaw=angle,
                    trav_map_og_size=size,
                    trav_map_og_res=resolution,
                    configs=RIG_config
                )

        print("\nPOMDP: ")
        print(pomdp.pretty_str())
        
if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    main(cfg)
