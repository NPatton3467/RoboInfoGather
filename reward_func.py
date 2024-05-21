import numpy as np
import json
from RoboInfoGather.MCTS_planner import Action
from RoboInfoGather.map_utils import *
from RoboInfoGather.observation_utils import get_fov

class RewardFunc():
    def __init__(self, map_params, camera_params, rf_params):
        self.map_params = map_params
        self.camera_params = camera_params
        self.rf_params = rf_params

    def eval(self, belief, obstacle_map, root, node):
        # If not taking an observation action return 0
        if node.inbound_act != Action.OBS:
            return 0

        # Get grid spaces within robots FOV
        local_config = {'rf_params': {'angle_delta' : self.rf_params['angle_delta']}}
        locs = get_fov(node.loc, local_config, self.camera_params, obstacle_map, belief, debug_print=False)

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
