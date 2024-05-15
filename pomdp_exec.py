import os

import yaml

import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options

from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image

from omnigibson.object_states.pose import Pose

from RoboInfoGather.MCTS_planner import *
from RoboInfoGather.observation_utils import *


def planned_action_to_real_action(act):
    # Translate planned discrete action into joint vel

    # ACTIONS
    # Format: OrderedDict([('robot0', [vel , angular_vel])])
    # Example (Spin CCW) : OrderedDict([('robot0', [0 , 1])])
    # Example (Spin CW) : OrderedDict([('robot0', [0 , -1])])
    # Example (Forward) : OrderedDict([('robot0', [1 , 0])])
    # Example (Backward) : OrderedDict([('robot0', [-1 , 0])])

    # CCW Rotation
    if act == Action.R_CCW:
        return OrderedDict([('robot0', [0 , 1])])
    
    # CW Rotation
    if act == Action.R_CW:
        return OrderedDict([('robot0', [0 , -1])])

    # Move Forward
    if act == Action.M_FORWARD:
        return OrderedDict([('robot0', [1 , 0])])

    # Move Backward
    if act == Action.M_BACKWARD:
        return OrderedDict([('robot0', [-1 , 0])])

    #Observation
    if act == Action.OBS:
        return OrderedDict([('robot0', [0 , 0])])

def MCTS_planner_exec(pomdp, obstacle_map, configs, pos, ori):
    # Instantiate new planner
    # Can't reuse old tree since info is probably not relevant anymore????
    start = Loc(pos[0], pos[1], np.arccos(quat_to_rot(ori)[0][0]))
    planner = MCTS_Planner(start, pomdp, obstacle_map, configs)

    best_next_node = planner.search()

    print("MCTS Action: ", best_next_node.inbound_act)
    print("MCTS Total Reward/Vists: ", best_next_node.total_rewards, best_next_node.visits)

    return best_next_node.loc

def low_level_planner_exec(way_point, pos, ori, config):
    reached_way_point = False
    angle = np.arccos(quat_to_rot(ori)[0][0])

    dist_to_waypoint = np.sqrt((way_point.x - pos[0])**2 + (way_point.y - pos[1])**2)
    if dist_to_waypoint < config['planner_params']['way_point_loc_acc']:
        if (angle - way_point.theta) < config['planner_params']['way_point_ang_acc']:
            reached_way_point = True
            action = OrderedDict([('robot0', [0 , 0])])

        else: # Need to rotate to face correct direction
            if angle > way_point.theta:
                # Rotate CCW
                action = OrderedDict([('robot0', [0 , 1])])
            else:
                # Rotate CW
                action = OrderedDict([('robot0', [0 , -1])])
             
    else: # Need to head towards waypoint
        # Get the angle towards the way point and check if we need to rotate to
        # face that direction

        delta_x = way_point.x - pos[0]
        delta_y = way_point.y - pos[1]

        delta_ang = np.arctan(delta_y/delta_x)

        if abs(delta_ang) < config['planner_params']['way_point_direction_angle']:
            # Drive towards waypoint
            action = OrderedDict([('robot0', [1 , 0])])
        else: # Need to rotate to face waypoint
            if delta_ang > way_point.theta:
                # Rotate CCW
                action = OrderedDict([('robot0', [0 , 1])])
            else:
                # Rotate CW
                action = OrderedDict([('robot0', [0 , -1])])


    return action, reached_way_point

def pomdp_exec_loop(env, pomdp, obstacle_map, config, dino_model):
    """
    Main loop for pomdp_execution
    """

    reached_way_point = True
    time_steps_since_MCTS = 0

    # Run until complete
    done, symbolic_info = pomdp.enough_info()
    print("In POMDP Exec loop -- DONE?: ", done)
    while not done:
        pos, ori = env.robots[0].get_position_orientation()
        print("Robot Angle: ", np.arccos(quat_to_rot(ori)[0][0]))

        # Get next action
        if reached_way_point or time_steps_since_MCTS > config['planner_params']['max_time_wo_replan']:
            print("Entering MCTS Planner")
            way_point = MCTS_planner_exec(pomdp, obstacle_map, config, pos, ori)
            time_steps_since_MCTS = 0

        print("Entering Low Level Planner, Waypoint: ", way_point.x, way_point.y)
        action, reached_way_point = low_level_planner_exec(way_point, pos, ori, config)
        time_steps_since_MCTS += 1
        
        print("Executing: ", action)
        state, _, _, _ = env.step(action)

        camera_pos, camera_ori = env.robots[0]._sensors['robot0:eyes:Camera:0'].get_position_orientation()

        # Update Obstacle map 
        lidar_sensor = env.robots[0]._sensors['robot0:scan_link:Lidar:0']
        print(state['robot0'].keys())
        scan = state['robot0']['robot0:scan_link:Lidar:0']['scan']

        obstacle_map.update(lidar_sensor, scan)

        obstacle_map.visualize()

        # Update POMDP
            # 1. Loop over all object types
            # 2. Get all visible voxels for that object types resolution
            # 3. Get predicted value (obs)
            # 4. Call update for that object types belief
        for obj_tp in pomdp.bel.keys():
            # Get predictions for all voxels based on observations
            vox_preds = get_vox_preds(camera_pos, camera_ori, pomdp.bel[obj_tp], obj_tp, state['robot0'], dino_model, config, obstacle_map)
            pomdp.bel[obj_tp].update(vox_preds)


            # Do the same for each feature
            for feature in pomdp.bel[obj_tp].feature_bels.keys():
                # Get predictions for all voxels based on observations
                vox_preds = get_vox_preds(camera_pos, camera_ori, pomdp.bel[obj_tp], obj_tp, state['robot0'], dino_model, config, obstacle_map, feature)
                pomdp.bel[obj_tp].update(vox_preds, feature=feature)


            plt.imshow(pomdp.bel[obj_tp].get_visualization())
            plt.show()

        # Check if Done
        done, symbolic_info = pomdp.enough_info()

    return symbolic_info
