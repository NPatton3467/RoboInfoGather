from pomdp_exec import *
from groundingdino.util.inference import load_model
from program_utils import *
from pomdp import *
from obstacle_map import *

import torch

#Temp until I get more GPU resources
torch.cuda.is_available = lambda : False

import argparse

def main(nl):
    """
    Main loop for info gathering. Given a natural language query, generate a program in our dsl.
    Then foreach query, generate a POMDP for the query and execute it.
    After executing all queries, evaluate their symbolic results and return the desired information

    :param: (nl) Natural language query representing the information to be gathered

    :return: (query_results)
    """

    # Load the config
    config_filename = os.path.join(f"./RoboInfoGather/info_gather.yaml")
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

    # Generate program
    prog = gen_prog_from_nl(nl)

    # Set up models
    CONFIG_PATH = os.path.join("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    WEIGHTS_PATH = os.path.join("./GroundingDINO/weights/groundingdino_swint_ogc.pth")
    dino_model = load_model(CONFIG_PATH, WEIGHTS_PATH)


    # Setup obstacle map
    # Make default trav_map size
    resolution = config['scene']['trav_map_resolution']
    trav_map = get_trav_map(config['scene']['trav_map_path'], config['scene']['floor'], resolution, resolution)

    trav_map = np.array(trav_map)
    trav_map[:, 0] = 128
    print(trav_map)
    plt.imshow(trav_map)
    plt.show()

    # Change lidar mounting
    _, rob_ori = env.robots[0].get_position_orientation()
    cur_scan_pos, _ =env.robots[0]._sensors['robot0:scan_link_Lidar_sensor'].get_position_orientation()
    cur_scan_pos[2] += 0.1
    env.robots[0]._sensors['robot0:scan_link_Lidar_sensor'].set_position_orientation(cur_scan_pos, rob_ori)

    size, _ = trav_map.shape
    obstacle_map = ObstacleMap(resolution, size)

    # Execute each query
    query_results = []
    for query in prog.expressions:
        pos, ori = env.robots[0].get_position_orientation()
        pomdp = gen_pomdp_from_query(query=query, pos=pos, ori=ori, trav_map_og_size=size, trav_map_og_res=resolution, configs=config)

        symbolic_info = pomdp_exec_loop(env, pomdp, obstacle_map, config, dino_model)

        query_results.append(query.execute(symbolic_info))

    # Always close the environment at the end
    env.close()

    return query_reults


if __name__ == '__main__':
    parser = argparse.ArgumentParser('info_gather_main')
    parser.add_argument('nl_query', type=str)

    args = parser.parse_args()

    nl = args.nl_query

    results = main(nl)