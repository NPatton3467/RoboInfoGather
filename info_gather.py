from pomdp_exec import *
from groundingdino.util.inference import load_model
from program_utils import *
from pomdp import *
from obstacle_map import *

import torch

#Temp until I get more GPU resources
#torch.cuda.is_available = lambda : False

import argparse
import pickle
import time

def main(nl):
    """
    Main loop for info gathering. Given a natural language query, generate a program in our dsl.
    Then foreach query, generate a POMDP for the query and execute it.
    After executing all queries, evaluate their symbolic results and return the desired information

    :param: (nl) Natural language query representing the information to be gathered

    :return: (query_results)
    """

    # Load the config
    config_filename = os.path.join(f"/robodata/user_data/npatt/OmniGibson/RoboInfoGather/info_gather.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # check if we want to quick load or full load the scene
    load_options = {
        "Quick": "Only load the building assets (i.e.: the floors, walls, doors)",
        "Full": "Load all interactive objects in the scene",
    }
    load_mode = choose_from_options(options=load_options, name="load mode", random_selection=False)
    if load_mode == "Quick":
        config["scene"]["load_object_categories"] = ["floors", "walls", "door"]

    # Load the environment
    env = og.Environment(configs=config)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # Reset env before start? 
    og.log.info("Resetting environment")
    env.reset()


    # Setup obstacle map
    # Make default trav_map size
    resolution = config['scene']['trav_map_resolution']
    trav_map = get_trav_map(config['scene']['trav_map_path'], config['scene']['floor'], resolution, resolution)

    trav_map = np.array(trav_map)
    trav_map[:, 0] = 128
    plt.imshow(trav_map)
    plt.show()

    # Change lidar mounting
    _, rob_ori = env.robots[0].get_position_orientation()
    print(env.robots[0]._sensors.keys())
    cur_scan_pos, _ =env.robots[0]._sensors['robot0:scan_link:Lidar:0'].get_position_orientation()
    cur_scan_pos[2] += 0.1
    env.robots[0]._sensors['robot0:scan_link:Lidar:0'].set_position_orientation(cur_scan_pos, rob_ori)

    # Change Camera Mounting
    camera_pos, camera_ori = env.robots[0]._sensors['robot0:eyes:Camera:0'].get_position_orientation()
    camera_pos[2] += 0.2
    env.robots[0]._sensors['robot0:eyes:Camera:0'].set_position_orientation(camera_pos, camera_ori)

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser('info_gather_main')
    parser.add_argument('nl_query', type=str)

    args, _ = parser.parse_known_args()

    dataset = args.dataset

    main(dataset)

    end_time = time.time()

    print(f'TIME: {end_time-start_time}s')
    print(f'SIM_TIME: {sim_time}s')
    
    # Always close the environment at the end
    env.close()
