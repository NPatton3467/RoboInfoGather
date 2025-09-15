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

    def eval(self, belief, obstacle_map, pt, angle):
        # Get grid spaces within robots FOV
        local_config = {'rf_params': {'angle_delta' : self.rf_params['angle_delta'],
                                        'dist_delta': self.rf_params['dist_delta']}}
        locs= get_fov(pt, angle, local_config, self.camera_params, obstacle_map, belief, debug_print=False)

        reward = 0.0
        x_max = belief.map_params['dim'][0]
        y_max = belief.map_params['dim'][1]
        z_max = belief.map_params['dim'][2]
        checked_xyz = []
        num_checked = 0
        for (x,y,z) in locs:
            vol_origin = belief.map_params['vol_origin']
            b_res = belief.map_params['res']
            bz_res = belief.map_params['z_res']
            map_dim = belief.map_params['dim']
            bxyz = world_to_map(np.array([x, y, z]), vol_origin, b_res, bz_res, map_dim)

            # Skip if bordeline out of range
            if bxyz[0] not in range(0, x_max) or bxyz[1] not in range(0, y_max) or\
                    bxyz[2] not in range(0, z_max):
                continue

            if (bxyz[0], bxyz[1], bxyz[2]) in checked_xyz:
                continue
            else:
                checked_xyz.append((bxyz[0], bxyz[1], bxyz[2]))

            for z in range(map_dim[2]):
                p = belief.p[bxyz[0], bxyz[1], bxyz[2]].detach().cpu()
                if p > 0.0 and p < 1.0:
                    num_checked += 1
                    reward += p * np.log(p) + (1-p)*np.log(1-p)

        # TEMP
        #reward = -1*reward/max(1, num_checked)
        reward = -1*reward
        # END TEMP

        return reward
