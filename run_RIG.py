"""
Run EQA in OmniGibson with RoboInfoGather exploration.

"""

# General Tool Imports
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
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
from scipy.spatial.transform import Rotation

# RoboInfoGather Imports
from RoboInfoGather.program_utils import *
from RoboInfoGather.MCTS_planner import *
from RoboInfoGather.map_utils import *
from RoboInfoGather.info_gather_runner import *

# Simulator Imports
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options

# Don't use GPU dynamics and use flatcache for performance boost
from omnigibson.macros import gm
gm.HEADLESS = True
gm.REMOTE_STREAMING="webrtc"
gm.OMNIGIBSON_REMOTE_STREAMING="webrtc"
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

# VLM Imports
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from src.geom import get_cam_intr
import importlib
VLM = importlib.import_module('explore-eqa.src.vlm').VLM


def load_models(cfg):
    """
    Load the VLM models 
    vlm: Prismatic VLM used in next ranking next waypoints from image
    molmo_tools: Molmo model and processor used for object detection
    """

    # Load VLM 
    vlm = VLM(cfg.vlm)
    vlm.model._supports_cache_class = False

    # Load Molmo
    # Set up MOLMO
    processor = AutoProcessor.from_pretrained(
        #'allenai/Molmo-72B-0924',
        #'allenai/MolmoE-1B-0924',
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    molmo_model = AutoModelForCausalLM.from_pretrained(
        #'allenai/Molmo-72B-0924',
        #'allenai/MolmoE-1B-0924',
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    molmo_tools = {'model': molmo_model, 'processor': processor}

    vlm_models = {'vlm': vlm, 'molmo_tools': molmo_tools}
    
    return vlm_models

def load_dataset(cfg):
    """
    Load all of the questions based on the dataset specified in the config
    """

    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]

    return questions_data

def setup_environmet(cfg, questions_data, question_ind):
    # Load the environment
    print("Question?\n", questions_data[question_ind])
    scene_name = questions_data[question_ind]['scene']
    scene_config_file = cfg.scene_config_path + f"{scene_name}.yaml"
    scene_config = yaml.load(open(scene_config_file, "r"), Loader=yaml.FullLoader)
    env = og.Environment(configs=scene_config)

    resolution = scene_config['scene']['trav_map_resolution']
    trav_map = get_trav_map(scene_config['scene']['trav_map_path'], scene_config['scene']['floor'], resolution, resolution)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Reset env before start? 
    og.log.info("Resetting environment")
    env.reset()


    # Change lidar mounting
    _, rob_ori = env.robots[0].get_position_orientation()
    print(env.robots[0]._sensors.keys())
    cur_scan_pos, _ =env.robots[0]._sensors['rob:scan_link:Lidar:0'].get_position_orientation()
    cur_scan_pos[2] += 0.1
    env.robots[0]._sensors['rob:scan_link:Lidar:0'].set_position_orientation(cur_scan_pos, rob_ori)

    # Change Camera Mounting
    camera_pos, camera_ori = env.robots[0]._sensors['rob:eyes:Camera:0'].get_position_orientation()
    camera_pos[2] += 0.2
    env.robots[0]._sensors['rob:eyes:Camera:0'].set_position_orientation(camera_pos, camera_ori)
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_width, img_height = env.robots[0]._sensors['rob:eyes:Camera:0'].camera_parameters["renderProductResolution"]
    cam_intr = np.array(env.robots[0]._sensors['rob:eyes:Camera:0'].intrinsic_matrix.detach().cpu())

    camera_data = {
            'camera_pos': camera_pos,
            'camera_ori': camera_ori,
            'camera_tilt': camera_tilt,
            'img_data': {'w': img_width, 'h': img_height},
            'cam_intr': cam_intr
        }

    # Floor - use pts height as floor height
    floor_height = 0.01
    map_size = trav_map.shape[0]
    tsdf_bnds = np.array(
            [
                [(-map_size/2.0)*resolution, (map_size/2.0)*resolution],
                [(-map_size/2.0)*resolution, (map_size/2.0)*resolution],
                [floor_height - 0.2, floor_height + 3.5]
            ])
    scene_size = (map_size * resolution) ** 2
    num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)*3 #TODO: REMOVE 3x
    logging.info(
        f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
    )

    scene_data = {
            'floor_height': floor_height,
            'map_size': map_size,
            'tsdf_bnds': tsdf_bnds,
            'scene_size': scene_size,
            'num_step': num_step,
            'debug_f_path': setup_debug_dir(cfg, question_ind)
        }
    
    # Get initial points and angle
    position_data = init_position_data(env)

    return env, camera_data, scene_data, position_data

def setup_debug_dir(cfg, question_ind):
    debug_f_path = cfg.debug_path + f"{question_ind}/"
    if not os.path.isdir(debug_f_path):
        os.mkdir(debug_f_path)

    return debug_f_path

def extract_task_info(questions_data, question_ind):
    question_data = questions_data[question_ind]
    scene = question_data["scene"]
    floor = question_data["floor"]
    scene_floor = scene + "_" + floor
    question = question_data["question"]
    # This was first but doesn't work choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
    choices = question_data["choices"].split("',")
    answer = question_data["answer"]
    text_answer = choices[0]
    if answer == "B":
        text_answer = choices[1]
    elif answer == "C":
        text_answer = choices[2]
    elif answer == "D":
        text_answer = choices[3]

    text_answer = text_answer.lstrip("'\\([ ")
    text_answer = text_answer.rstrip("'\\)] ")
    
    logging.info(f"\n========\nIndex: {question_ind} Scene: {scene} Floor: {floor}")

    # Set data dir for this question - set initial data to be saved
    episode_data_dir = os.path.join(cfg.output_dir, str(question_ind))
    os.makedirs(episode_data_dir, exist_ok=True)
    result = {"question_ind": question_ind}

    task_info = {
            'question': question,
            'text_answer': text_answer,
            'episode_data_dir': episode_data_dir
        }

    return task_info

def init_position_data(env):
    init_pts, _ = env.robots[0].get_position_orientation()
    init_pts = np.array(init_pts.cpu().detach())
    pitch, roll, init_angle = env.robots[0].get_rpy()
    init_angle = init_angle.cpu().detach()

    position_data = {
            'pts': init_pts,
            'angle': init_angle,
            'pitch': pitch,
            'roll': roll
        }

    return position_data

def save_all_data(cfg, results_all, cnt_data, cum_sim_score, question_ind):
    with open(os.path.join(cfg.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results_all, f)

    logging.info(f"\n== All Summary")
    logging.info(f"Number of data collected: {cnt_data}")
    
    print("Cumulative Sim Score: ", cum_sim_score)
    print("Current Number of Questions: ", question_ind+1)
    print("LLM-Match %: ", cum_sim_score/(question_ind+1)*100)

def main(cfg):
    # Load dataset
    questions_data = load_dataset(cfg)
   
    # Load Models
    vlm_models = load_models(cfg) 

    # Run all questions
    cnt_data = 0
    results_all = []
    cum_sim_score = 0
    for question_ind in tqdm(range(len(questions_data))):
        # Setup environment
        env, camera_data, scene_data, position_data = setup_environment(cfg, questions_data, question_ind)

        # Get task information
        task_info = extract_task_info(questions_data, question_ind)

        ###################################
        # Run Info Gathering for the Task #
        ###################################
        result = info_gather_runner(
                cfg,
                env,
                camera_data,
                scene_data,
                task_info
                cum_sim_score,
                cnt_data,
                position_data,
                vlm_models
            )
        
        # Save data
        results_all.append(result)
        cnt_data += 1

        og.clear()


    # Save all data again
    save_all_data(cfg, results_all, cnt_data, cum_sim_score, question_ind)

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(cfg.output_dir, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)
