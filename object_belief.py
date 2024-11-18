import numpy as np
import json

from RoboInfoGather.map_utils import *

class ObjTpBel():
    def __init__(self, num, threshold, map_params, configs, relevant_features=None):
        self.num = num # Num objects to be found, If None -> unbounded
        self.threshold = threshold # Existence threshold
        self.map_params = map_params
        print("PATH: ", configs['scene']['trav_map_path'])
        print("FLOOR: ", configs['scene']['floor'])
        print("RES: ", map_params['res'])
        print("OG RES: ", map_params['og_res'])
        self.trav_map = get_trav_map(configs['scene']['trav_map_path'], configs['scene']['floor'], map_params['res'], map_params['og_res'])
        self.configs = configs
        self.relevant_features = relevant_features
        
        #self.p = np.copy(self.trav_map)

        self.p = np.ones_like(self.trav_map)
        self.p = self.p * 0.5
        self.p = np.where(self.trav_map == 0, -1, self.p)

        print(f'Belief Created with (xdim, y_dim) = ({self.p.shape})')

        # Start with uniform prior, where traversable
        #self.p = np.where((self.p == 255), 0.5, 0)

        # Need to replicate vertically
        self.z_dim = int(self.configs['rf_params']['map_height'] / self.map_params['z_res'])
        temp_p = [self.p for i in range(self.z_dim)]
        self.p = np.stack(temp_p, axis=2)

        print('Belief shape: ', self.p.shape)

        # For copying later if we get new features to evaluate
        self.backup_p = np.copy(self.p)

        # For object detection matching
        self.clusters = []

        # Belief over features
        self.feature_bels = {}
        if self.relevant_features != None and self.relevant_features != [None]:
            for feature in self.relevant_features:
                if feature['name'] != None:
                    if feature['tp'] == "feature_scalar":
                        feature_dict = {'bel': np.copy(self.p), "tp" : feature['tp']}
                        self.feature_bels[feature['name']] = feature_dict
                    elif feature['tp'] == "feature_enum":
                        # For features, we want to keep around names like colour = red
                        x_dim, y_dim, z_dim = self.p.shape
                        bel_z = np.array(['' for _ in range(z_dim)], dtype=object)
                        bel_y = np.array([bel_z for _ in range(y_dim)], dtype=object)
                        bel = np.array([bel_y for _ in range(x_dim)], dtype=object)

                        # Should have same shape
                        assert bel.shape == self.p.shape


                        feature_dict = {'bel': bel, "tp" : feature['tp']}
                        self.feature_bels[feature['name']] = feature_dict
                    else:
                        assert False # Shouldn't get here



    def update(self, obs, eps=1e-6, feature=None):
        if feature is None:
            # Update Belief using Binary Bayes Filter
            log_p = np.where(self.p > 0, np.log((self.p+eps)/(1-self.p+eps)), 0)

            print("p min/max", np.min(self.p), "/", np.max(self.p))
            print("obs min/max", np.min(obs), "/", np.max(obs))

            inv_sensor_model = np.where(obs != -1, np.log((obs+eps)/(1-obs+eps)), 0)

            #new_log_p = np.where(inv_sensor_model != 0, log_p + inv_sensor_model, log_p)
            new_log_p = log_p + inv_sensor_model

            self.p = 1 - (1/(1+np.exp(new_log_p)))

            if np.isnan(self.p).any():
                print("Belief:")
                print(self.p)
                print("Observation:")
                print(obs)
                assert False
            
            del(inv_sensor_model)
            del(new_log_p)
            del(log_p)

            extended_trav_map = np.expand_dims(self.trav_map, axis=-1)
            extended_trav_map = np.tile(extended_trav_map, (1,1, self.p.shape[2]))
            self.p = np.where(extended_trav_map == 0, -1, self.p)

            print("Done Update")
        else:
            print("Feature: ", feature)
            assert False # This needs to change to reflect aribitrary features (e.g. colour will have values of "red" here)
            p = self.feature_bels[feature]

            log_p = np.log((p+eps)/(1-p+eps))

            inv_sensor_model = np.where(obs != -1, np.log((obs+eps)/(1-obs+eps)), 0)

            new_log_p = np.where(inv_sensor_model != 0, log_p + inv_sensor_model, log_p)

            self.feature_bels[feature] = 1 - (1/(1+np.exp(new_log_p)))

    def get_visualization(self):
        return np.mean(self.p, axis=2)
