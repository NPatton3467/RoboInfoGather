import numpy as np
import json
from RoboInfoGather.MCTS_planner import Action
from RoboInfoGather.map_utils import *

class RewardFunc():
    def __init__(self, map_params, camera_params, rf_params):
        self.map_params = map_params
        self.camera_params = camera_params
        self.rf_params = rf_params

    def get_new_node(self, current_loc, dist, angle):
        # In robot frame: robot direction is X-axis.

        # Find the X,Y locations of the point in robot frame 
        # from distance and angle
        new_x = np.cos(angle) * dist
        new_y = np.sin(angle) * dist

        # Do a rotation based on robot theta to get delta x,y in real coords
        delta_x = new_x * np.cos(current_loc.theta) - new_y * np.sin(current_loc.theta)
        delta_y = new_x * np.sin(current_loc.theta) + new_y * np.cos(current_loc.theta)

        # Get actual x and y based off of current loc
        real_x = current_loc.x + delta_x
        real_y = current_loc.y + delta_y

        # Get Map xy
        return real_x, real_y

    def get_robot_fov(self, node, obstacle_map):
        min_angle = self.camera_params['min_angle']
        max_angle = self.camera_params['max_angle']
        min_v_dist = self.camera_params['min_visual_distance']
        max_v_dist = self.camera_params['max_visual_distance']

        angle_delta = self.rf_params['angle_delta']
        dist_delta = self.rf_params['dist_delta']

        angle = min_angle

        fov = []
        while angle < max_angle:
            dist = min_v_dist
            while dist < max_v_dist:
                # Get node that corresponds to angle and dist (relative to robot)
                x, y = self.get_new_node(node.loc, dist, angle)

                # Skip if already added
                if (x, y) in fov:
                    dist += dist_delta
                    continue

                # Check that x,y are within map bounds
                res = obstacle_map.resolution
                size = obstacle_map.size
                x_max = size
                y_max = size
                mxy = world_to_map(np.array([x, y]), map_resolution=res, map_size=size)
                if mxy[0] not in range(0, x_max) or mxy[1] not in range(0, y_max):
                    break

                # Can't see through obstacles so break
                # TODO: Think about how to represent with different map
                # granularities once LIDAR setup
                if obstacle_map.obstacles[mxy[0], mxy[1]]:
                    break

                fov.append((x,y))

                # Increment distance
                dist += dist_delta

            # Increment angle
            angle += angle_delta

        return fov

    def eval(self, belief, obstacle_map, root, node):
        # If not taking an observation action return 0
        if node.inbound_act != Action.OBS:
            return 0

        # Get grid spaces within robots FOV
        locs = self.get_robot_fov(node, obstacle_map)

        reward = 0.0
        x_max = belief.map_params['size']
        y_max = belief.map_params['size']
        for (x,y) in locs:
            bxy = world_to_map(np.array([x, y]), belief.map_params['res'], belief.map_params['size'])

            # Skip if bordeline out of range
            if bxy[0] not in range(0, x_max) or bxy[1] not in range(0, y_max):
                continue

            for z in range(belief.z_dim):
                p = belief.p[bxy[0], bxy[1], z]
                if p > 0.0 and p < 1.0:
                    reward += p * np.log(p) + (1-p)*np.log(1-p)

        reward *= -1

        # Add distance to get observation
        # root_x = root.loc.x
        # root_y = root.loc.y

        # node_x = node.loc.x
        # node_y = node.loc.y

        # dist = np.sqrt((node_x-root_x)**2 + (node_y-root_y)**2)

        return reward
