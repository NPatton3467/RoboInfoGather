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
        if (node.inbound_act != Action.OBS ):# and
            #node.inbound_act != Action.L_N and
            #node.inbound_act != Action.L_NE and
            #node.inbound_act != Action.L_E and
            #node.inbound_act != Action.L_SE and
            #node.inbound_act != Action.L_S and
            #node.inbound_act != Action.L_SW and
            #node.inbound_act != Action.L_W and
            #node.inbound_act != Action.L_NW):
           return 0

        # Get grid spaces within robots FOV
        local_config = {'rf_params': {'angle_delta' : self.rf_params['angle_delta']}}
        #locs = get_fov(node.loc, local_config, self.camera_params, obstacle_map, belief, debug_print=False)
        locs, _ = get_fov(node.loc, local_config, self.camera_params, obstacle_map, belief, debug_print=False)

        reward = 0.0
        x_max = belief.map_params['size']
        y_max = belief.map_params['size']
        checked_xy = []
        num_checked = 0
        for (x,y) in locs:
            bxy = world_to_map(np.array([x, y]), belief.map_params['res'], belief.map_params['size'])

            # Skip if bordeline out of range
            if bxy[0] not in range(0, x_max) or bxy[1] not in range(0, y_max):
                continue

            if (bxy[0], bxy[1]) in checked_xy:
                continue
            else:
                checked_xy.append((bxy[0], bxy[1]))

            for z in range(belief.z_dim):
                p = belief.p[bxy[0], bxy[1], z].detach().cpu()
                if p > 0.0 and p < 1.0:
                    num_checked += 1
                    reward += p * np.log(p) + (1-p)*np.log(1-p)

        reward = -1*reward/max(1, num_checked)

        print('Num Checked: ', len(checked_xy))
        print('Actual Num Checked: ', num_checked)
        print('Num in FOV: ', len(locs))

        # Add distance to get observation
        # root_x = root.loc.x
        # root_y = root.loc.y

        # node_x = node.loc.x
        # node_y = node.loc.y

        # dist = np.sqrt((node_x-root_x)**2 + (node_y-root_y)**2)

        return reward
