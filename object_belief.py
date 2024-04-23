import numpy as np
import json

from RoboInfoGather.map_utils import *

class ObjTpBel():
    def __init__(self, num, threshold, map_params, configs, relevant_features=None):
        self.num = num # Num objects to be found, If None -> unbounded
        self.threshold = threshold # Existence threshold
        self.map_params = map_params
        self.trav_map = get_trav_map(configs['scene']['trav_map_path'], configs['scene']['floor'], map_params['res'], map_params['og_res'])
        self.configs = configs
        self.relevant_features = relevant_features
        
        self.p = np.copy(self.trav_map)

        print(f'Belief Created with (xdim, y_dim) = ({self.p.shape})')

        # Start with uniform prior, where traversable
        self.p = np.where((self.p == 255), 0.5, 0)

        # Need to replicate vertically
        self.z_dim = int(self.configs['rf_params']['map_height'] / self.map_params['res'])
        temp_p = [self.p for i in range(self.z_dim)]
        self.p = np.stack(temp_p, axis=2)

        print('Belief shape: ', np.shape(self.p))

        # For copying later if we get new features to evaluate
        self.backup_p = np.copy(self.p)

        # Belief over features
        self.feature_bels = {}
        if self.relevant_features != None and self.relevant_features != [None]:
            for feature in self.relevant_features:
                if feature['name'] != None:
                    feature_dict = {'bel': np.copy(self.p), "tp" : feature['tp']}
                    self.feature_bels[feature['name']] = feature_dict



    def update(self, obs, eps=1e-6, feature=None):
        """
        # Discretize voxels and map to belief ranges
        vxy = [voxel_x, voxel_y]
        bxy = world_to_map(vxy, self.map_params['res'], self.map_params['size'])
        bx = bxy[0]
        by = bxy[1]
        bz = int(voxel_z / self.map_params['res'])

        # Check that in range
        if bx not in range(0, self.p.shape[0]):
            return

        if by not in range(0, self.p.shape[1]):
            return

        if bz not in range(0, self.p.shape[2]):
            return


        # Check that locaiton is not a wall
        if self.trav_map[bx, by, bz] != 255:
            return

        print(f'Updating at Index = ({bx}, {by}, {bz}), with likelihood = {obs}')"""

        if feauture is None:
            # Update Belief using Binary Bayes Filter
            # CAN THIS BE DONE WITHOUT DOING EACH VOXEL INDIVIDUALLY? -- yes
            p = self.p

            log_p = np.log(p/(1-p))

            inv_sensor_model = np.where(obs != 0, np.log((obs-eps)/(1-obs+eps)), 0)

            new_log_p = np.where(inv_sensor_model != 0, log_p + inv_sensor_model, log_p)

            self.p = 1 - (1/(1+np.exp(new_log_p)))
        else:
            p = self.feature_bels[feature]

            log_p = np.log(p/(1-p))

            inv_sensor_model = np.where(obs != 0, np.log((obs-eps)/(1-obs+eps)), 0)

            new_log_p = np.where(inv_sensor_model != 0, log_p + inv_sensor_model, log_p)

            self.feature_bels[feature] = 1 - (1/(1+np.exp(new_log_p)))