import os
import pickle

import yaml

#import omnigibson as og
#from omnigibson.utils.ui_utils import choose_from_options

from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import queue

#from omnigibson.object_states.pose import Pose

from RoboInfoGather.MCTS_planner import *
from RoboInfoGather.observation_utils import *
from RoboInfoGather.map_utils import *
from RoboInfoGather.motion_planning import *

from dataclasses import dataclass, field
from typing import Any

from scipy.spatial.transform import Rotation

def eval_reached_way_point(goal_pos, goal_yaw, pos, yaw, config):
    dist = np.sqrt((goal_pos[0] - pos[0])**2 + (goal_pos[1] - pos[1])**2)
    if dist < config['planner_params']['way_point_loc_acc']:
        if np.abs(yaw - goal_yaw) < config['planner_params']['way_point_ang_acc'] or\
                abs(goal_yaw- yaw) > (2*np.pi - config['planner_params']['way_point_direction_angle']):

                return True

    return False


def get_goal(way_point):
    cur_node = way_point
    while len(cur_node.children) > 0:
        best_child = cur_node.children[0]
        highest_reward = best_child.total_rewards

        for child in cur_node.children:
            if child.total_rewards > highest_reward:
                highest_reward = child.total_rewards
                best_child = child

        cur_node = best_child

    return np.array([cur_node.loc.x, cur_node.loc.y]), cur_node.loc.theta

def get_path(start_pos, child_node, obstacle_map):
    # Use A* to find path to terminal node
    cur_node = child_node
    while len(cur_node.children) > 0:
        best_child = cur_node.children[0]
        highest_reward = best_child.visits

        for child in cur_node.children:
            if child.total_rewards > highest_reward:
                highest_reward = child.total_rewards
                best_child = child

        cur_node = best_child

    start_loc = world_to_map(np.array([start_pos[0], start_pos[1]]), obstacle_map.resolution, obstacle_map.size)
    goal_loc = world_to_map(np.array([cur_node.loc.x, cur_node.loc.y]), obstacle_map.resolution, obstacle_map.size)
    #path = a_star(goal_loc, start_loc, obstacle_map)

    return -1, cur_node.loc.theta


def sample_locs(start, pomdp, obstacle_map, configs):
    best_reward = 0
    best_sample_node = None
    best_astar_path = []

    num_checked = 0


    while num_checked < configs['planner_params']['max_num_samples']:
        # Sample Inflated Obstalce map for map coords
        o_size = obstacle_map.size
        o_res = obstacle_map.resolution
        s_disc = configs['planner_params']['sample_discretization']
        x_obs_map_coords = np.random.randint(0, int(o_size/s_disc)) * s_disc
        y_obs_map_coords = np.random.randint(0, int(o_size/s_disc)) * s_disc

        xy_world_coords = map_to_world(np.array([x_obs_map_coords, y_obs_map_coords]), o_res, o_size)

        # Check if legal in belief
        legal = True
        for key in pomdp.bel.keys():
            b_res = pomdp.bel[key].map_params['res']
            b_size = pomdp.bel[key].map_params['size']
            xy_bel_coords = world_to_map(xy_world_coords, b_res, b_size)

            if xy_bel_coords[0] >= pomdp.bel[key].p.shape[0]:
                legal = False
                break
        
            if xy_bel_coords[1] >= pomdp.bel[key].p.shape[1]:
                legal = False
                break

            if (pomdp.bel[key].p[xy_bel_coords[0], xy_bel_coords[1]] == -1).any():
                legal = False
                break

        if not legal:
            continue

        # Check if feasible path
        state = {'pos': np.array([start.x, start.y])}
        goal_pos = xy_world_coords
        path = a_star(goal_pos, state, obstacle_map)

        if path == []:
            continue

        # Check reward
        root = None
        t_s_disc = configs['planner_params']['angle_sample_discretization']
        theta = np.deg2rad(np.random.randint(0, int(360/t_s_disc)) * t_s_disc)
        pred_goal_loc = Loc(goal_pos[0], goal_pos[1], theta)
        node = MCTS_Tree_Node(
                loc=pred_goal_loc,
                obstacle_map = obstacle_map,
                num_prev_obs = 0,
                max_obs = 1,
                config = configs,
                inbound_act = Action.OBS
                )
        reward = 0
        for key in pomdp.bel.keys():
            belief = pomdp.bel[key]
            reward += pomdp.reward_funcs[key].eval(belief, obstacle_map, root, node)

        if reward > best_reward:
            best_reward = reward
            best_sample_node = node
            best_astar_path = path

        num_checked += 1

    print("Done Sample Locs")
    print("Best Loc: ", best_sample_node.loc.x, best_sample_node.loc.y, best_sample_node.loc.theta)
    print("Best Reward: ", best_reward)
    return best_sample_node, best_astar_path

def MCTS_planner_exec(pomdp, obstacle_map, configs, pos, yaw, iteration):
    # Instantiate new planner
    # Can't reuse old tree since info is probably not relevant anymore????
    start = Loc(pos[0], pos[1], yaw)

    max_map_dim = obstacle_map.size * obstacle_map.resolution
    #max_rollout_depth = int(np.sqrt(2 * (max_map_dim ** 2)) * 1.5 / configs['planner_params']['mcts_step_length']) + 1
    #max_rollout_depth = int(np.sqrt(2 * (max_map_dim ** 2)) * 1.5 / 0.25) + 1
    max_rollout_depth = configs['planner_params']['max_rollout_depth']

    #planner = MCTS_Planner(start, pomdp, obstacle_map, configs, max_rollout_depth=max_rollout_depth)

    #best_next_node = planner.search()

    best_next_node, astar_path = sample_locs(start, pomdp, obstacle_map, configs)

    print("MCTS Action: ", best_next_node.inbound_act)
    print("MCTS Loc: ", best_next_node.loc.x, best_next_node.loc.y, best_next_node.loc.theta)
    print("MCTS Total Reward/Vists: ", best_next_node.total_rewards, best_next_node.visits)

    # Debug -- print belief, cur_loc and new_loc
    if True:
        def get_last_node(node, res, size):
            cur_node = node
            while len(cur_node.children) > 0:
                best_child = cur_node.children[0]
                highest_reward = best_child.visits

                for child in cur_node.children:
                    if child.total_rewards > highest_reward:
                        highest_reward = child.total_rewards
                        best_child = child

                cur_node = best_child
            x = cur_node.loc.x
            y = cur_node.loc.y
            t = cur_node.loc.theta
            return world_to_map(np.array([cur_node.loc.x, cur_node.loc.y]), res, size), x,y,t

        for key in pomdp.bel.keys():
            res = pomdp.bel[key].map_params['res']
            size = pomdp.bel[key].map_params['size']
            cur_loc_map_coords =  world_to_map(np.array([start.x, start.y]), res, size)
            goal_loc_map_coords, x,y,t = get_last_node(best_next_node, res, size)
            
            b = np.copy(pomdp.bel[key].p)

            b = np.mean(b, axis=2)
            
            b[cur_loc_map_coords[0], cur_loc_map_coords[1]] = 2
            b[goal_loc_map_coords[0], goal_loc_map_coords[1]] = 3

            # For belief readability
            b = np.where(b == -1, 0, b)

            #plt.close()
            #heatmap = plt.pcolor(b)
            #plt.colorbar(heatmap)
            #plt.gca().invert_yaxis()
            #plt.show()

            with open(f'/robodata/user_data/npatt/OmniGibson/debug/MCTS/{iteration}.pkl', 'wb') as f:
                pickle.dump(pomdp, f)

            start_arr = np.array([start.x, start.y, yaw])
            goal_arr = np.array([x,y,t])
            
            np.save(f'/robodata/user_data/npatt/OmniGibson/debug/MCTS/start_{iteration}.npy', start_arr)
            np.save(f'/robodata/user_data/npatt/OmniGibson/debug/MCTS/goal_{iteration}.npy', goal_arr)


    return best_next_node, astar_path

"""
def low_level_planner_exec(path, pos, yaw, goal_yaw, config):

    print("In Low-level planner, path: ", path)

    reached_way_point = False
    angle = yaw 

    if len(path) > 1:
        way_point = Loc(path[0][0], path[0][1], yaw)
    else:
        way_point = Loc(path[0][0], path[0][1], goal_yaw)

    print(way_point)

    dist_to_waypoint = np.sqrt((way_point.x - pos[0])**2 + (way_point.y - pos[1])**2)
    if dist_to_waypoint < config['planner_params']['way_point_loc_acc']:
        if np.abs(angle - way_point.theta) < config['planner_params']['way_point_ang_acc'] or\
                abs(way_point.theta - angle) > (2*np.pi - config['planner_params']['way_point_direction_angle']):

            path.pop(0)
            action = OrderedDict([('robot0', [0 , 0])])
            if len(path) == 0:
                reached_way_point = True

        else: # Need to rotate to face correct direction
            rot_vel = abs(way_point.theta - angle) / np.deg2rad(15)
            if (angle < way_point.theta and abs(way_point.theta - angle) < np.pi) or\
                    (angle > way_point.theta and abs(way_point.theta - angle) > np.pi):
                # Rotate CCW
                action = OrderedDict([('robot0', [0 , 0.15*rot_vel])])
            else:
                # Rotate CW
                action = OrderedDict([('robot0', [0 , -0.15 * rot_vel])])
             
    else: # Need to head towards waypoint
        # Get the angle towards the way point and check if we need to rotate to
        # face that direction

        delta_x = way_point.x - pos[0]
        delta_y = way_point.y - pos[1]

        delta_ang = np.arctan(delta_y/delta_x)

        # Account for angles > 90, < -90
        if delta_x < 0:
            if delta_y < 0:
                delta_ang = delta_ang - np.pi
            else:
                delta_ang = delta_ang + np.pi

        if abs(delta_ang - angle) < config['planner_params']['way_point_direction_angle'] or\
                abs(delta_ang - angle) > (2*np.pi - config['planner_params']['way_point_direction_angle']):
            # Drive towards waypoint
            action = OrderedDict([('robot0', [dist_to_waypoint , 0])])
        else: # Need to rotate to face waypoint
            rot_vel = abs(delta_ang - angle) / np.deg2rad(15)
            if (angle < delta_ang and abs(delta_ang - angle) < np.pi) or\
                    (angle > delta_ang and abs(delta_ang - angle) > np.pi):
                # Rotate CCW
                action = OrderedDict([('robot0', [0 , rot_vel])])
            else:
                # Rotate CW
                action = OrderedDict([('robot0', [0 , -1 * rot_vel])])


    return action, reached_way_point, path

"""

def pomdp_exec_loop(env, pomdp, obstacle_map, config, dino_model):
    """
    Main loop for pomdp_execution
    """
    action = OrderedDict([('robot0', [0 , 0])])
    state, _, _, info = env.step(action)

    #while True:
    #    action = OrderedDict([('robot0', [0, 1])])
    #    state, _, _, info = env.step(action)

    #    print("Angular Velocity: ", env.robots[0].get_angular_velocity())
    #    print("Velocity: ", env.robots[0].get_linear_velocity())
    

    camera_pos, camera_ori = env.robots[0]._sensors['robot0:eyes:Camera:0'].get_position_orientation()
    camera_rpy = env.robots[0]._sensors['robot0:eyes:Camera:0'].get_rpy()
    # Offset camera angle correctly
    camera_rpy[2] += np.deg2rad(90)
    camera_intrinsic_mat = env.robots[0]._sensors['robot0:eyes:Camera:0'].intrinsic_matrix
    camera_params = env.robots[0]._sensors['robot0:eyes:Camera:0'].camera_parameters

    proj = np.array(env.robots[0]._sensors['robot0:eyes:Camera:0'].camera_parameters['cameraProjection'])
    proj = np.reshape(proj, (4,4))
    view = np.array(env.robots[0]._sensors['robot0:eyes:Camera:0'].camera_parameters['cameraViewTransform'])
    view = np.reshape(view, (4,4))
    print("Proj: ", proj)
    print("View: ", view)
    proj_view = np.matmul(view,proj)
    print("View Proj Mat: ", proj_view)

    pos, ori = env.robots[0].get_position_orientation()
    pitch, roll, yaw = env.robots[0].get_rpy()
    print("Robot Pos/Angle: ", pos, yaw)

    og_pitch, og_roll, yaw = env.robots[0].get_rpy()

    # Update Obstacle map 
    lidar_sensor = env.robots[0]._sensors['robot0:scan_link:Lidar:0']
    print(state['robot0'].keys())
    scan = state['robot0']['robot0:scan_link:Lidar:0']['scan']

    """
    obs, info = env.robots[0]._sensors['robot0:eyes:Camera:0']._get_obs()

    print("OBS: ", obs)
    print("INFO: ", info)

    print(info['seg_instance'])

    straight_chair_amgwaw_0_id = -1
    for inst_id in info['seg_instance']:
        if info['seg_instance'][inst_id] == 'straight_chair_amgwaw_0':
            straight_chair_amgwaw_0_id = inst_id
            break
   
    temp_seg = state['robot0']['robot0:eyes:Camera:0']['seg_instance']
    temp_seg = np.where(temp_seg == straight_chair_amgwaw_0_id, 1, 0)
    plt.imshow(temp_seg)
    plt.show()

    # Make bbox (xyxy)
    y, x = np.where(temp_seg == 1)

    bbox = (np.min(x), np.min(y), np.max(x), np.max(y))
    print(bbox)

    for x in range(bbox[0], bbox[2]):
        temp_seg[bbox[1], x] = 2
        temp_seg[bbox[3], x] = 2

    for y in range(bbox[1], bbox[3]):
        temp_seg[y, bbox[0]] = 2
        temp_seg[y, bbox[2]] = 2

    plt.imshow(temp_seg)
    plt.show()

    assert False
    """
    
    obstacle_map.update(lidar_sensor, scan)
    
    iterations = 0

    # Update POMDP
        # 1. Loop over all object types
        # 2. Get all visible voxels for that object types resolution
        # 3. Get predicted value (obs)
        # 4. Call update for that object types belief
    found_obj = False
    pitch, roll, yaw = env.robots[0].get_rpy()
    for obj_tp in pomdp.bel.keys():
        # Get predictions for all voxels based on observations
        vox_preds, found_obj = get_vox_preds(yaw, camera_pos, camera_ori, camera_rpy, pomdp.bel[obj_tp], obj_tp, state['robot0'], info, dino_model, config, obstacle_map, camera_intrinsic_mat, camera_params, iteration=iterations)
        pomdp.bel[obj_tp].update(vox_preds)
            
        if found_obj:
            found_obj = False


        # Do the same for each feature
        for feature in pomdp.bel[obj_tp].feature_bels.keys():
            # Get predictions for all voxels based on observations
            vox_preds, found_obj = get_vox_preds(yaw, camera_pos, camera_ori, camera_rpy, pomdp.bel[obj_tp], obj_tp, state['robot0'], info, dino_model, config, obstacle_map, camera_intrinsic_mat, camera_params, feature, iteration=iterations)
            pomdp.bel[obj_tp].update(vox_preds, feature=feature)

    reached_way_point = True
    way_point = None
    time_steps_since_MCTS = 0

    time_steps_no_movement = 0
    last_pos = None

    planned_path = []

    replans = 0

    # Run until complete
    done, symbolic_info = pomdp.enough_info(iterations)
    print("In POMDP Exec loop -- DONE?: ", done)

    sim_time = 0

    while not done:
        if iterations != 0:
            done, symbolic_info = pomdp.enough_info(iterations)
            if done:
                return symbolic_info, sim_time
        
        pos, ori = env.robots[0].get_position_orientation()
        pitch, roll, yaw = env.robots[0].get_rpy()
        print("Robot Pos/Angle: ", pos, yaw)
        

        # Check if stuck
        if last_pos is not None and np.sqrt((last_pos[0] - pos[0]) ** 2 + (last_pos[1] - pos[1]) ** 2) < 0.01:
            time_steps_no_movement += 1
        else:
            time_steps_no_movement = 0
            last_pos = pos
        
        if time_steps_no_movement > 10:
            # Send some zeros to stop movement
            i = 0
            while i < 5:
                i += 1
                action = OrderedDict([('robot0', [-1 , 0])])
                print("Executing: ", action)
                start_step_time = time.time()
                state, _, _, info = env.step(action)
                end_step_time = time.time()

                sim_time += (end_step_time - start_step_time)
            
            time_steps_no_movement = 0
            last_pos = pos

        # Get next action
        if reached_way_point or time_steps_since_MCTS > config['planner_params']['max_time_wo_replan']:
            replans = 0
            next_carrot = None
            # Check if Done
            done, symbolic_info = pomdp.enough_info(iterations)
            if done:
                return symbolic_info, sim_time

            # Send some zeros to stop movement
            i = 0
            while i < 50:
                i += 1
                action = OrderedDict([('robot0', [0 , 0])])
                print("Executing: ", action)
                start_step_time = time.time()
                state, _, _, info = env.step(action)
                end_step_time = time.time()

                sim_time += (end_step_time - start_step_time)

            pos = env.robots[0].get_position()
            pitch, roll, yaw = env.robots[0].get_rpy()
            print("Robot Pos/Angle: ", pos, yaw)

            print("Entering MCTS Planner")
            way_point, astar_path = MCTS_planner_exec(pomdp, obstacle_map, config, pos, yaw, iterations)

            planned_path, goal_yaw = get_path(pos, way_point, obstacle_map)

            time_steps_since_MCTS = 0

        if planned_path != None and planned_path != []:
            if config['planner_params']['teleport']:
                # Get Goal
                cur_node = way_point
                while len(cur_node.children) > 0:
                    best_child = cur_node.children[0]
                    highest_reward = best_child.total_rewards / best_child.visits

                    for child in cur_node.children:
                        if (child.total_rewards / best_child.visits) > highest_reward:
                            highest_reward = best_child.total_rewards / best_child.visits
                            best_child = child

                    cur_node = best_child

                # Get next a star waypoint to teleport to
                max_found_dist = 0
                max_idx = 0
                max_dist = 0.5
                best_point = None

                idx = 0
                for point in astar_path:
                    dist = np.sqrt((pos[0] - point[0]) ** 2 + (pos[1] - point[1])**2)
                    if dist < max_dist and idx >= max_idx and dist > max_found_dist:
                        max_found_dist = dist
                        max_idx = idx
                        best_point = point

                # Check if reached end waypoint
                if point[0] == cur_node.loc.x and point[1] == cur_node.loc.y:
                    reached_waypoint = True

                goal_node = np.array([point[0], point[1]])
                goal_yaw = np.arctan2(pos[1] - goal_node[1], pos[0] - goal_node[0])

                print("Teleporting to: ", goal_node[0], goal_node[1])
                print("Yaw: ", goal_yaw)
                action = OrderedDict([('robot0', [0, 0])])
                pitch, roll, yaw = env.robots[0].get_rpy()
                rot_to_send = Rotation.from_euler('xyz', [og_pitch, og_roll, goal_yaw], degrees=False)
                ori_to_send = rot_to_send.as_quat()
                env.robots[0].set_position_orientation([goal_node[0], goal_node[1], pos[2]], ori_to_send)
                #reached_way_point = True
            else:
                if planned_path == -1:
                    planned_path = None
                else:
                    print("Entering Low Level Planner, Next Waypoint: ", planned_path[0][0], planned_path[0][1])
                    print("Length of Planned Path: ", len(planned_path))
                
                print("Full Angular Velocity: ", env.robots[0].get_angular_velocity())
                goal_pos, goal_yaw = get_goal(way_point)
                motion_state = {
                    'yaw': yaw,
                    'pos': np.array([pos[0], pos[1]]),
                    'lin_vel': env.robots[0].get_linear_velocity(),
                    'ang_vel': env.robots[0].get_angular_velocity()[2],
                    'max_vel_diff': 0.25,
                    'max_curv_diff': 1.0,
                    'max_decel' : 1
                }
                action, planned_path, replans, action_buffer, next_carrot = motion_plan_main(
                        config=config,
                        state=motion_state,
                        goal_pos=goal_pos,
                        goal_yaw=goal_yaw,
                        obstacle_map=obstacle_map,
                        action_buffer=[],
                        carrot_radius=0.2,
                        next_carrot=next_carrot,
                        path=planned_path,
                        replans=replans
                )
                #action, reached_way_point, planned_path = low_level_planner_exec(planned_path, pos, yaw, goal_yaw, config)
            time_steps_since_MCTS += 1

            if not config['planner_params']['teleport']:
                reached_way_point = eval_reached_way_point(goal_pos, goal_yaw, pos, yaw, config)
        else:
            action = OrderedDict([('robot0', [0 , 0])])
            reached_way_point = True
            
            time_steps_no_movement = 0
            last_pos = pos
        

        print("Executing: ", action)
        start_step_time = time.time()
        state, _, _, info = env.step(action)
        
        # TEMP #####
        state, _, _, info = env.step(action)
        state, _, _, info = env.step(action)
        state, _, _, info = env.step(action)
        state, _, _, info = env.step(action)
        end_step_time = time.time()

        sim_time += (end_step_time - start_step_time)

        camera_pos, camera_ori = env.robots[0]._sensors['robot0:eyes:Camera:0'].get_position_orientation()
        camera_rpy = env.robots[0]._sensors['robot0:eyes:Camera:0'].get_rpy()
        # Offset camera angle correctly
        camera_rpy[2] += np.deg2rad(90)
        camera_intrinsic_mat = env.robots[0]._sensors['robot0:eyes:Camera:0'].intrinsic_matrix
        camera_params = env.robots[0]._sensors['robot0:eyes:Camera:0'].camera_parameters

        # Update Obstacle map 
        lidar_sensor = env.robots[0]._sensors['robot0:scan_link:Lidar:0']
        print(state['robot0'].keys())
        scan = state['robot0']['robot0:scan_link:Lidar:0']['scan']

        obstacle_map.update(lidar_sensor, scan)

        pitch, roll, yaw = env.robots[0].get_rpy()

        # Update POMDP
            # 1. Loop over all object types
            # 2. Get all visible voxels for that object types resolution
            # 3. Get predicted value (obs)
            # 4. Call update for that object types belief
        found_obj = False
        for obj_tp in pomdp.bel.keys():
            # Get predictions for all voxels based on observations
            vox_preds, found_obj = get_vox_preds(yaw, camera_pos, camera_ori, camera_rpy, pomdp.bel[obj_tp], obj_tp, state['robot0'], info, dino_model, config, obstacle_map, camera_intrinsic_mat, camera_params, iteration=iterations)
            pomdp.bel[obj_tp].update(vox_preds)
        
            if found_obj:
                found_obj = False
                np.save(f'/robodata/user_data/npatt/OmniGibson/debug/updated_beliefs/{obj_tp}_{iterations}.npy', pomdp.bel[obj_tp].p)


            # Do the same for each feature
            for feature in pomdp.bel[obj_tp].feature_bels.keys():
                # Get predictions for all voxels based on observations
                vox_preds, found_obj = get_vox_preds(yaw, camera_pos, camera_ori, camera_rpy, pomdp.bel[obj_tp], obj_tp, state['robot0'], info, dino_model, config, obstacle_map, camera_intrinsic_mat, camera_params, feature, iteration=iterations)
                pomdp.bel[obj_tp].update(vox_preds, feature=feature)


            if iterations % 100 == 0:
                obstacle_map.visualize()

                visualization = pomdp.bel[obj_tp].get_visualization()


                res = pomdp.bel[obj_tp].map_params['res']
                size = pomdp.bel[obj_tp].map_params['size']
                rp_xy = world_to_map(np.array([pos[0], pos[1]]), res, size)

                visualization[rp_xy[0], rp_xy[1]] = 0.95

                plt.imshow(visualization)
                plt.savefig('cur_bel.png')

        iterations += 1
        found_obj = False

        print("Done Exec Loop: ", iterations)

    return symbolic_info, sim_time
