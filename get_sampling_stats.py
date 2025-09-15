import pickle
import argparse
import os
import yaml
import json
import copy

import numpy as np
from scipy.spatial.transform import Rotation

from collections import OrderedDict
    
# RoboInfoGather Import
from map_utils import *

import multiprocessing
from multiprocessing import Process, Queue

# Global mapping from tasks to relevant objects
objs_per_task = [
    ['plant'],
    ['cabinet'],
    ['cabinet', 'plant'],
    ['chair', 'table'],
    ['cabinet'],
    ['cabinet', 'plant'],
    ['chair', 'table'],
]

def get_average_min_distance(result, res_dir, task_idx, scene_config):
    # Setup hash for object instances
    min_distance = {}
    for obj in objs_per_task[task_idx]:
        min_distance[obj] = {}

    # Get the JSON file for the scene to get object positions
    with open(scene_config['scene']['scene_file'], 'rb') as f:
        og_scene = json.load(f)

    # Iterate through each step and check for minimum occlusions of each relevant object instance
    for i in range(50):
        with open(res_dir + f'debug/{task_idx}/info_{i}.npy', 'rb') as f:
            info = np.load(f, allow_pickle=True)[()]
            seg_inst_mapping = info['obs_info']['rob']['rob:eyes:Camera:0']['seg_instance']

        # Iterate through each object type and check for instances
        for obj in min_distance:
            for obj_inst in seg_inst_mapping:
                # If instance found check in segmentation image
                if seg_inst_mapping[obj_inst].find(obj) >= 0:
                    # Check if instance is already hashed
                    inst_name = seg_inst_mapping[obj_inst]

                    robot_pos = result[f'step_{i}']['pts']
                    obj_pos = og_scene['state']['object_registry'][inst_name]['root_link']['pos']

                    dist = np.linalg.norm(robot_pos - obj_pos)

                    if obj_inst in min_distance[obj]:
                        # Check if current min is smaller
                        if min_distance[obj][inst_name] > dist:
                            min_distance[obj][inst_name] = dist

                    else:
                        min_distance[obj][inst_name] = dist

    # Average Over max Pixels
    print(min_distance)
    dsum = 0
    num = 0
    for obj in min_distance:
        for inst in min_distance[obj]:
            dsum += min_distance[obj][inst]
            num += 1

    return dsum/num


def get_average_max_pixels(res_dir, task_idx):
    # Setup hash for object instances
    max_pixels = {}
    for obj in objs_per_task[task_idx]:
        max_pixels[obj] = {}

    # Iterate through each step and check for minimum occlusions of each relevant object instance
    for i in range(50):
        with open(res_dir + f'debug/{task_idx}/info_{i}.npy', 'rb') as f:
            info = np.load(f, allow_pickle=True)[()]
            seg_inst_mapping = info['obs_info']['rob']['rob:eyes:Camera:0']['seg_instance']

        with open(res_dir + f'debug/{task_idx}/seg_inst_{i}.npy', 'rb') as f:
            seg_inst = np.load(f)

        # Iterate through each object type and check for instances
        for obj in max_pixels:
            for obj_inst in seg_inst_mapping:

                # If instance found check in segmentation image
                if seg_inst_mapping[obj_inst].find(obj) >= 0:
                    # Get number of pixels that correspond to this instance
                    num_pix = np.sum(np.where(seg_inst == obj_inst, 1, 0))

                    # Check if instance is already hashed
                    inst_name = seg_inst_mapping[obj_inst]
                    if obj_inst in max_pixels[obj]:
                        # Check if current max is larger
                        if max_pixels[obj][inst_name] < num_pix:
                            max_pixels[obj][inst_name] = num_pix

                    else:
                        max_pixels[obj][inst_name] = num_pix

    # Average Over max Pixels
    print(max_pixels)
    psum = 0
    num = 0
    for obj in max_pixels:
        for inst in max_pixels[obj]:
            psum += max_pixels[obj][inst]
            num += 1

    return psum/num



def get_unoccluded_num_pix(queue, result, seg_inst_name, scene_config, time_step):
    # Get scene_config json file and modify it to only include this object instance
    with open(scene_config['scene']['scene_file'], 'rb') as f:
        og_scene = json.load(f)

    #print(og_scene.keys()) # ['metadata', 'state', 'objects_info']
    #print(og_scene['metadata'].keys()) # NO SUB KEYS
    #print(og_scene['state'].keys()) # dict_keys(['system_registry', 'object_registry'])
    #print(og_scene['state']['system_registry'].keys()) # NO SUB KEYS
    #print(og_scene['state']['object_registry'].keys()) # OBJECT INSTANCES
    #print(og_scene['objects_info'].keys()) # dict_keys(['init_info'])
    #print(og_scene['objects_info']['init_info'].keys()) # OBJECT INSTANCES

    temp_scene = {
            'metadata' : {},
            'state': {
                    'system_registry': {},
                    'object_registry': {
                            seg_inst_name : og_scene['state']['object_registry'][seg_inst_name],
                            "floors_avvnuv_0" : og_scene['state']['object_registry']["floors_avvnuv_0"],
                            "floors_gjemfr_0" : og_scene['state']['object_registry']["floors_gjemfr_0"],
                            "floors_ifmioj_0" : og_scene['state']['object_registry']["floors_ifmioj_0"],
                            "floors_ptwlei_0" : og_scene['state']['object_registry']["floors_ptwlei_0"],
                            "floors_ucjdgj_0" : og_scene['state']['object_registry']["floors_ucjdgj_0"],
                        },
                },
            'objects_info': {
                    'init_info': {
                            seg_inst_name : og_scene['objects_info']['init_info'][seg_inst_name],
                            "floors_avvnuv_0" : og_scene['objects_info']['init_info']["floors_avvnuv_0"],
                            "floors_gjemfr_0" : og_scene['objects_info']['init_info']["floors_gjemfr_0"],
                            "floors_ifmioj_0" : og_scene['objects_info']['init_info']["floors_ifmioj_0"],
                            "floors_ptwlei_0" : og_scene['objects_info']['init_info']["floors_ptwlei_0"],
                            "floors_ucjdgj_0" : og_scene['objects_info']['init_info']["floors_ucjdgj_0"],
                        },
                },
            }


    # Save temp scene and use to load omnigibson simulator
    with open('temp.json', 'w') as f:
        json.dump(temp_scene, f)

    new_scene_config = copy.deepcopy(scene_config)
    new_scene_config['scene']['scene_file'] = './temp.json'

    # Start Env
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

    env = og.Environment(configs=new_scene_config)
    # Reset env before start?
    og.log.info("Resetting environment")
    env.reset()


    # Change lidar mounting
    _, rob_ori = env.robots[0].get_position_orientation()
    cur_scan_pos, _ =env.robots[0]._sensors['rob:scan_link:Lidar:0'].get_position_orientation()
    cur_scan_pos[2] += 0.1
    env.robots[0]._sensors['rob:scan_link:Lidar:0'].set_position_orientation(cur_scan_pos, rob_ori)

    # Change Camera Mounting
    # TEMP
    env.robots[0]._sensors['rob:eyes:Camera:0'].horizontal_aperture = 50
    env.robots[0]._sensors['rob:eyes:Camera:0'].camera_parameters["cameraAperture"][0] = 50
    # END TEMP

    camera_pos, camera_ori = env.robots[0]._sensors['rob:eyes:Camera:0'].get_position_orientation()
    camera_pos[2] += 0.2
    env.robots[0]._sensors['rob:eyes:Camera:0'].set_position_orientation(camera_pos, camera_ori)

    # Set position and orientation from result point and get the observation
    roll, pitch, _ = env.robots[0].get_rpy()
    pts = result[f'step_{time_step}']['pts']
    angle = result[f'step_{time_step}']['angle']

    # Set position
    ori_to_send = Rotation.from_euler('xyz', [pitch, roll, angle], degrees=False).as_quat()
    while np.isnan(ori_to_send).any():
        angle += 1
        ori_to_send = Rotation.from_euler('xyz', [pitch, roll, angle], degrees=False).as_quat()

    env.robots[0].set_position_orientation(pts, ori_to_send)
    action = OrderedDict([('rob', np.array([0 , 0]))])
    state, _, _, _, info = env.step(action) # Take Empty step to get observations
    state, _, _, _, info = env.step(action) # Take Empty step to get observations
    state, _, _, _, info = env.step(action) # Take Empty step to get observations

    og.clear()
    env = None

    # Remove temp JSON
    os.remove('temp.json')

    # Get seg_inst
    seg_inst = state['rob']['rob:eyes:Camera:0']['seg_instance']
    inst_id = -1
    for inst in info['obs_info']['rob']['rob:eyes:Camera:0']['seg_instance']:
        if seg_inst_name == info['obs_info']['rob']['rob:eyes:Camera:0']['seg_instance'][inst]:
            inst_id = inst

    assert inst_id >= 0

    num_pix = np.sum(np.where(np.array(seg_inst.detach().cpu()) == inst_id, 1, 0))

    queue.put(num_pix)


def estimate_min_occlusion(result, res_dir, scene_config, task_idx):
    # Setup hash for object instances
    min_occlusions = {}
    for obj in objs_per_task[task_idx]:
        min_occlusions[obj] = {}

    # Iterate through each step and check for minimum occlusions of each relevant object instance
    for i in range(50):
        with open(res_dir + f'debug/{task_idx}/info_{i}.npy', 'rb') as f:
            info = np.load(f, allow_pickle=True)[()]
            seg_inst_mapping = info['obs_info']['rob']['rob:eyes:Camera:0']['seg_instance']

        with open(res_dir + f'debug/{task_idx}/seg_inst_{i}.npy', 'rb') as f:
            seg_inst = np.load(f)

        # Iterate through each object type and check for instances
        for obj in min_occlusions:
            for obj_inst in seg_inst_mapping:

                # If instance found check in segmentation image
                if seg_inst_mapping[obj_inst].find(obj) >= 0:
                    # Get number of pixels that correspond to this instance
                    num_pix = np.sum(np.where(seg_inst == obj_inst, 1, 0))

                    multiprocessing.set_start_method('spawn', force=True)
                    q = Queue()
                    p = Process(target=get_unoccluded_num_pix,
                            args=(q, result, seg_inst_mapping[obj_inst], scene_config, i))
                    p.start()
                    p.join()
                    unoccluded_num_pix = q.get()

                    print(f"\n\n\n\nUNOCCLUDED PIXEL COUNT: {unoccluded_num_pix}\n\n\n\n")

                    # Check if instance is already hashed
                    inst_name = seg_inst_mapping[obj_inst]
                    if obj_inst in min_occlusions[obj]:
                        # Check if current min is smaller
                        if min_occlusions[obj][inst_name] > (num_pix / unoccluded_num_pix):
                            min_occlusions[obj][inst_name] = num_pix / unoccluded_num_pix

                    else:
                        min_occlusions[obj][inst_name] = num_pix / unoccluded_num_pix

    return min_occlusions


def trav_map_to_world(xy, map_resolution, map_size):
    axis = 0 if len(xy.shape) == 1 else 1
    return np.flip((xy - map_size / 2.0) * map_resolution, axis=axis)

def estimate_coverage(trav_map, trav_map_res, trav_map_size, tsdf, bels):
    legal_pts = np.argwhere(trav_map == 255)

    unexplored = (np.sum(tsdf._explore_vol_cpu, axis=-1) == 0).astype(int)

    voxels_covered = 0
    for pt in legal_pts:
        # Get world coord and check if explored
        world_coord = trav_map_to_world(pt, trav_map_res, trav_map_size)
        # I THINK THIS ONE IS WRONG? world_coord = trav_map_to_world(np.array([pt[1], pt[1]]), trav_map_res, trav_map_size)
        tsdf_coord = tsdf.world2vox(np.array([world_coord[0], world_coord[1], 0]))

        explored_in_bel_space = False
        for obj_tp in bels:
            map_resolution = bels[obj_tp].map_params['res']
            map_dim = bels[obj_tp].map_params['dim']
            vol_origin = bels[obj_tp].map_params['vol_origin']
            z_resolution = bels[obj_tp].map_params['z_res']
            bel_coord = world_to_map(np.array([world_coord[0],world_coord[1],0]),
                    vol_origin,
                    map_resolution,
                    z_resolution,
                    map_dim)

            if (bels[obj_tp].p[bel_coord[0], bel_coord[1], :] != 0.5).any():
                explored_in_bel_space = True
                break

        if not unexplored[tsdf_coord[0], tsdf_coord[1]]: # or explored_in_bel_space:
            voxels_covered += 1

    return voxels_covered / legal_pts.shape[0]


if __name__ == '__main__':
    # Get results directory
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--res_dir", help="Results Directory", default="", type=str)
    args = parser.parse_args()
    res_dir = args.res_dir

    # Loop through results files
    total_coverage = 0
    num_checked = 0
    min_occlusions = []
    average_max_pixels = []
    average_min_distance = []
    for i in range(35):
        # Check if results_{i}.pkl exists
        if not os.path.exists(res_dir + f'results_{i}.pkl'):
            break

        num_checked += 1

        # Load results file
        with open(res_dir + f'results_{i}.pkl', 'rb') as f:
            result = pickle.load(f)

        # Get trav map to estimage coverage
        scene_config_file = './RoboInfoGather/configs/scene_configs/' + "Rs_int.yaml"
        scene_config = yaml.load(open(scene_config_file, "r"), Loader=yaml.FullLoader)
        resolution = scene_config['scene']['trav_map_resolution']
        trav_map_path = scene_config['scene']['trav_map_path']
        trav_map = get_trav_map(trav_map_path, scene_config['scene']['floor'], resolution, resolution)

        # Correct for weird ""traversable space"" that isn't actually traversable
        trav_map[:200, :200] = 0

        trav_map_size = trav_map.shape[0]
        trav_map_res = resolution
        tsdf = result['tsdf_planner']
        bels = result['pomdp'].bel
        total_coverage += estimate_coverage(trav_map, trav_map_res, trav_map_size, tsdf, bels) 

        #min_occlusions.append(estimate_min_occlusion(result, res_dir, scene_config, i))
        #average_max_pixels.append(get_average_max_pixels(res_dir, i))

        average_min_distance.append(get_average_min_distance(result, res_dir, i, scene_config))


    average_coverage = total_coverage / num_checked
    print("Average Coverage: ", average_coverage)

    print("Min Occlusions: ", min_occlusions)

    print("Average Max Pixels List: ", average_max_pixels)
    print("Average Max Pixels Average: ", np.sum(average_max_pixels)/len(average_max_pixels))

    print("Average Min Distance List: ", average_min_distance)
    print("Average Min Distance: ", np.sum(average_min_distance)/len(average_min_distance))
