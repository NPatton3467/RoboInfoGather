from dataclasses import dataclass, field
from typing import Any
import queue

from collections import OrderedDict

import numpy as np
import cv2

from RoboInfoGather.map_utils import *

from matplotlib import pyplot as plt

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any=field(compare=False)

class AStarNode():
    def __init__(self, g, f, h, loc, parent=None):
        self.g = g
        self.f = f
        self.h = h
        self.loc = loc
        self.parent = parent

def get_children(loc, obstacle_map, size):
    x_diff = [-1, 0, 1]
    y_diff = [-1, 0, 1]

    children = []
    for xd in x_diff:
        for yd in y_diff:
            if xd == 0 and yd == 0:
                continue

            nx = loc[0] + xd
            ny = loc[1] + yd

            if nx >= size or nx < 0 or \
                    ny >= size or ny < 0:
                    continue
            if obstacle_map[nx, ny]:
                continue

            children.append(np.array([nx, ny]))

    return children

def a_star(goal_pos, state, obstacle_map):
    open_list = queue.PriorityQueue()
    closed_list = []

    dilate_rad = int(np.ceil(0.45 / obstacle_map.resolution))
    local_obs_map = cv2.dilate(obstacle_map.obstacles, np.ones((dilate_rad,dilate_rad)))
    new_res = 0.1
    new_size = int((obstacle_map.size * obstacle_map.resolution) / new_res)
    local_obs_map = cv2.resize(local_obs_map, (new_size, new_size))
   
    start_loc = world_to_map(np.array([state['pos'][0], state['pos'][1]]), new_res, new_size)
    goal_loc = world_to_map(np.array([goal_pos[0], goal_pos[1]]), new_res, new_size)

    goal_loc[0] = max(0, min(goal_loc[0], new_size))
    goal_loc[1] = max(0, min(goal_loc[1], new_size))

    if local_obs_map[goal_loc[0], goal_loc[1]]:
        return []

    start_node = AStarNode(0, 0, 0, start_loc, None)
    start_p_item = PrioritizedItem(0, start_node)
    open_list.put(start_p_item)

    num_iters = 0
    while not open_list.empty():# and num_iters < 10000:
        num_iters += 1
        cur_p_item = open_list.get()
        cur_node = cur_p_item.item

        if list(cur_node.loc) in [list(item) for item in closed_list]:
            continue

        print("Closing node: ", cur_node.loc)
        print("Goal Loc: ", goal_loc, " Goal Obstacle: ", local_obs_map[goal_loc[0], goal_loc[1]])
        print("Start Loc: ", start_loc)
        print("Closed List Length: ", len(closed_list))
        print("Closed List Max Length: ", new_size ** 2)
        closed_list.append(cur_node.loc)

        assert len(closed_list) < new_size ** 2

        if (cur_node.loc == goal_loc).all():
            print("Found Goal")
            path = []
            bt_node = cur_node
            while bt_node.parent != None:
                world_loc = map_to_world(np.array([bt_node.loc[0], bt_node.loc[1]]), new_res, new_size)
                map_loc = world_to_map(np.array([world_loc[0], world_loc[1]]), new_res, new_size)

                if map_loc[0] != bt_node.loc[0] or map_loc[1] != bt_node.loc[1]:
                    print(map_loc, bt_node.loc)
                    #assert False

                path.append(world_loc)

                local_obs_map[map_loc[0], map_loc[1]] = 1

                print(world_loc)

                bt_node = bt_node.parent

            world_loc = map_to_world(np.array([bt_node.loc[0], bt_node.loc[1]]), new_res, new_size)
            path.append(world_loc)
            path.reverse()

            print("Done A*, Showing PATH")
            #plt.imshow(local_obs_map)
            #plt.show()

            return path

        children = get_children(cur_node.loc, local_obs_map, new_size)

        for child in children:
            if list(child) in [list(item) for item in closed_list]:
                continue

            cost_to_go = cur_node.g + np.sqrt((child[0] - cur_node.loc[0])**2 + (child[1] - cur_node.loc[1])**2)
            heuristic = np.sqrt((goal_loc[0] - child[0])**2 + (goal_loc[1] - child[1])**2)
            f = cost_to_go + heuristic

            child_node = AStarNode(cost_to_go, f, heuristic, child, cur_node)

            child_p_item = PrioritizedItem(child_node.f, child_node)
            open_list.put(child_p_item)

    return []

def reached_next_carrot(next_carrot, pos, config):
    if euclid_dist(next_carrot, pos) < config['planner_params']['carrot_loc_acc']:
        print("\n\n\nSTART REACHED NEXT CARROT PRINTOUT")
        print("REACHED NEXT CARROT")
        print("REACHED NEXT CARROT")
        print("REACHED NEXT CARROT")
        print("REACHED NEXT CARROT")
        print("REACHED NEXT CARROT")
        print("END REACHED NEXT CARROT PRINTOUT\n\n\n")
        return True

    return False

def euclid_dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def invalid(path, cur_loc, carrot_radius):
    for point in path:
        if euclid_dist(cur_loc, point) < carrot_radius:
            return True

    return False


def get_next_carrot(cur_pos, goal_pos, carrot_radius, path):
    if euclid_dist(goal_pos, cur_pos) < carrot_radius:
        return goal_pos

    best_carrot = path[0]
    for point in path:
        if euclid_dist(point, cur_pos) < carrot_radius:
            best_carrot = point

    return best_carrot


def squared_norm(vec):
    return vec[0] ** 2 + vec[1] ** 2


def get_radial_max_free_path_length(cur_pos, cur_yaw, test_curv, local_target):
    turn_radius = 1 / test_curv
    quarter_circle_dist = abs(turn_radius) * np.pi / 2
    turn_center = np.array([0, turn_radius])
    target_radial = local_target - turn_center

    middle_radial = abs(turn_radius) * target_radial / np.sqrt(target_radial[0]**2 + target_radial[1]**2)
    middle_angle = np.arctan2(abs(middle_radial[0]), abs(middle_radial[1]))
    dist_closest_to_goal = middle_angle * abs(turn_radius)

    return min(dist_closest_to_goal, quarter_circle_dist)


def get_free_path_length(cur_pos, cur_yaw, next_carrot, obstacle_map, test_curv, state):
    # Get local target -- next carrot in robot frame
    neg_yaw_rot = np.array([[np.cos(-cur_yaw), -np.sin(-cur_yaw)],
                            [np.sin(-cur_yaw), np.cos(-cur_yaw)]])
    local_target = np.matmul(neg_yaw_rot, next_carrot - cur_pos)

    # Half length of robot + margin
    l = 0.3
    # Half width of robot plus margin
    w = 0.3

    # Check if moving straight ahead
    if test_curv < 1e-2 and test_curv > -1e-2:
        print("Straight Line Max Path Length Calculation")
        max_free_path_length = abs(local_target[0])
        print("Local Target x == initial max_free_path_length", max_free_path_length)

        # Check point cloud
        for point in obstacle_map.cur_world_scan:
            # Rotate and translate to robot frame
            point_2D = np.array([point[0], point[1]])
            p = np.matmul(neg_yaw_rot, point_2D - cur_pos)

            if abs(p[1]) > w:
                continue

            if abs(p[0]) < max_free_path_length:
                max_free_path_length = abs(p[0])
                print("Max fpl Updated: ", max_free_path_length)
            
            #max_free_path_length = min(max_free_path_length, abs(p[0]))

        vel = get_vel(max_free_path_length, state)

        print("Found Velocity: ", vel)

        if vel < 0:
            max_free_path_length = 0

        return max_free_path_length, vel

    else:
        path_radius = 1 / test_curv
        c = np.array([0, path_radius])
        s = 1
        if path_radius < 0:
            s = -1

        inner_front_corner = np.array([1, s * w])
        outer_front_corner = np.array([1, -s * w])

        r1 = max(0, abs(path_radius) - w)
        r1_sq = r1 ** 2
        r2_sq = squared_norm(inner_front_corner - c)
        r3_sq = squared_norm(outer_front_corner - c)

        angle_min = np.pi

        #print("Curve Max Path Length Calculation")
        max_free_path_length = get_radial_max_free_path_length(cur_pos, cur_yaw, test_curv, local_target)
        #print("Initial Max FPL: ", max_free_path_length)

        # Loop through point cloud
        for point in obstacle_map.cur_world_scan:
            # Rotate and translate to robot frame
            point_2D = np.array([point[0], point[1]])
            p = np.matmul(neg_yaw_rot, point_2D - cur_pos)

            if p[0] < 0:
                continue

            r_sq = squared_norm(p - c)
            if r_sq < r1_sq or r_sq > r3_sq:
                continue

            r = np.sqrt(r_sq)
            if test_curv > 0:
                theta = np.arctan2(p[0], path_radius - p[1])
            else:
                theta = np.arctan2(p[0], p[1] - path_radius)

            if r_sq < r2_sq:
                # Hits robot side first
                x = abs(path_radius) - w
                if x > 0:
                    alpha = np.arccos(x / r)
                else:
                    alpha = (np.pi / 2) + np.arccos(-x / r)
            else:
                # Hits robot front first
                alpha = np.arcsin(l / r)

            # TODO if max(0.0, abs(path_radius)*(theta-alpha)) < max_free_path_length:
                # TODO max_free_path_length = max(0.0, abs(path_radius)*(theta-alpha))

                #print("Updated Max FPL: ", max_free_path_length)

            #max_free_path_length = min(max_free_path_length, max(0.0, abs(path_radius)*(theta-alpha)))
        
        vel = get_vel(max_free_path_length, state)

        if vel < 0:
            max_free_path_length = 0

        return max_free_path_length, vel


def get_vel(free_path_length, state):
    vel = np.sqrt(squared_norm(state['lin_vel']))
    max_decel = state['max_decel']

    stopping_dist = (vel ** 2) / (2 * abs(max_decel))

    if stopping_dist > free_path_length:
        return -1

    else:
        test_vel = min(vel + state['max_vel_diff'], 1)
        stopping_dist = (test_vel ** 2) / (2 * abs(max_decel))
        if stopping_dist > free_path_length:
            return max(0, vel - state['max_vel_diff'])
        else:
            return test_vel


def get_local_action(next_carrot, state, action_buffer, obstacle_map):
    dilate_rad = int(np.ceil(0.45 / obstacle_map.resolution))
    inflated_obstacle_map = cv2.dilate(obstacle_map.obstacles, np.ones((dilate_rad,dilate_rad)))
    # Get Ackerman Vel
    cur_ack_vel = np.sqrt(state['lin_vel'][0]**2 + state['lin_vel'][1]**2)
    if state['yaw'] > np.deg2rad(0) and state['yaw'] < np.deg2rad(90) and \
            state['lin_vel'][0] < 0 and state['lin_vel'][1] < 0:
                cur_ack_vel *= -1
    if state['yaw'] > np.deg2rad(90) and state['yaw'] < np.deg2rad(180) and \
            state['lin_vel'][0] > 0 and state['lin_vel'][1] < 0:
                cur_ack_vel *= -1
    if state['yaw'] > np.deg2rad(180) and state['yaw'] < np.deg2rad(270) and \
            state['lin_vel'][0] > 0 and state['lin_vel'][1] > 0:
                cur_ack_vel *= -1
    if state['yaw'] > np.deg2rad(270) and state['yaw'] < np.deg2rad(360) and \
            state['lin_vel'][0] < 0 and state['lin_vel'][1] > 0:
                cur_ack_vel *= -1
    if state['yaw'] > np.deg2rad(-180) and state['yaw'] < np.deg2rad(-90) and \
            state['lin_vel'][0] > 0 and state['lin_vel'][1] > 0:
                cur_ack_vel *= -1
    if state['yaw'] > np.deg2rad(-90) and state['yaw'] < np.deg2rad(0) and \
            state['lin_vel'][0] < 0 and state['lin_vel'][1] > 0:
                cur_ack_vel *= -1

    cur_ack_curv = state['ang_vel'] / abs(cur_ack_vel)

    # Sample Linear Vel and Curvature
    max_curv_diff = state['max_curv_diff']

    curv_diffs = np.arange(-100, 101, 1) * (max_curv_diff / 100)

    # Find min distance to goal
    best_control_idx = 0
    best_dist = -1
    for i in range(len(curv_diffs)):
        # TODO: test_curv = cur_ack_curv + curv_diffs[i]
        test_curv = curv_diffs[i]

        state_2D = np.array([state['pos'][0], state['pos'][1]])
        free_path_length, vel = get_free_path_length(state_2D, state['yaw'], next_carrot, obstacle_map, test_curv, state)
        if free_path_length > best_dist:
            best_dist = free_path_length
            best_ack_vel = vel
            best_control_idx = i


    print("Best Free Path Length: ", best_dist)
    print("Best Velocity: ", best_ack_vel)

    if best_ack_vel != 0:
        # Convert lin_vel, curve to lin, twist
        best_curv = curv_diffs[best_control_idx]

        twist = best_ack_vel*best_curv
        action = OrderedDict([('robot0', [best_ack_vel, twist])])
    else:
        action = OrderedDict([('robot0', [0, 1])])

    return action, action_buffer

def motion_plan_main(config, state, goal_pos, goal_yaw, obstacle_map, action_buffer, carrot_radius, next_carrot, path=None, replans=0):
    start_loc = world_to_map(np.array([state['pos'][0], state['pos'][1]]), obstacle_map.resolution, obstacle_map.size)
    goal_loc = world_to_map(np.array([goal_pos[0], goal_pos[1]]), obstacle_map.resolution, obstacle_map.size)
    if path == None or path == [] or invalid(path, start_loc, carrot_radius):
        replans += 1
        path = a_star(goal_pos, state, obstacle_map)

        if path == []:
            next_action = OrderedDict([('robot0', [0, 0])])
            return next_action, path, replans, action_buffer, next_carrot

    dist = np.sqrt((goal_pos[0] - state['pos'][0])**2 + (goal_pos[1] - state['pos'][1])**2)
    if dist < config['planner_params']['way_point_loc_acc']:
        # Rotate in place
        next_action = OrderedDict([('robot0', [0, 1])])
        return next_action, path, replans, action_buffer, next_carrot


    # Get next carrot
    if next_carrot is None or reached_next_carrot(next_carrot, state['pos'], config):
        next_carrot = get_next_carrot(state['pos'], goal_pos, carrot_radius, path)

    print("Current Position: ", state['pos'])
    print("Current Velocity: ", state['lin_vel'])
    print("Current Angular Velocity: ", state['ang_vel'])
    print("Next Carrot: ", next_carrot)
    print("Goal Position: ", goal_pos)



    # Check if angle to next carrot is larger than 90 -- if so rotate in place
    delta_ang = np.arctan2(next_carrot[1] - state['pos'][1], next_carrot[0] - state['pos'][0])
    if abs(delta_ang - state['yaw']) > np.deg2rad(45):
        if state['yaw'] < delta_ang and abs(delta_ang - state['yaw']) < np.deg2rad(180):
            # Rotate in place
            next_action = OrderedDict([('robot0', [0, 1])])
            return next_action, path, replans, action_buffer, next_carrot
        else:
            # Rotate in place
            next_action = OrderedDict([('robot0', [0, -1])])
            return next_action, path, replans, action_buffer, next_carrot



    # Get action to next carrot
    next_action, action_buffer = get_local_action(next_carrot, state, action_buffer, obstacle_map)

    return next_action, path, replans, action_buffer, next_carrot
