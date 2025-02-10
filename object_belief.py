import numpy as np
import json

from RoboInfoGather.map_utils import *

class ObjTpBel():
    def __init__(self, num, threshold, map_params, configs, relevant_features=None):
        self.num = num # Num objects to be found, If None -> unbounded
        self.threshold = 0.75 # threshold # Existence threshold
        self.map_params = map_params

        # TODO: Will want to update trav map as we go
        size_to_use = max(2, map_params['size'])
        self.trav_map = np.zeros((size_to_use, size_to_use))

        self.configs = configs
        self.relevant_features = relevant_features
        
        #self.p = np.copy(self.trav_map)

        self.p = np.ones_like(self.trav_map)
        self.p = self.p * 0.5
        self.p = np.where(self.trav_map == 0, -1, self.p)

        # Start with uniform prior, where traversable
        #self.p = np.where((self.p == 255), 0.5, 0)

        # Need to replicate vertically
        self.z_dim = max(2, int(self.configs['rf_params']['map_height'] / self.map_params['z_res']))
        temp_p = [self.p for i in range(self.z_dim)]

        if self.z_dim > 1:
            self.p = np.stack(temp_p, axis=2)


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
                        feature_dict = {'bel': np.copy(self.p), "tp" : feature['tp'], "vals": np.zeros_like(self.p)}
                        self.feature_bels[feature['name']] = feature_dict
                    elif feature['tp'] == "feature_enum":
                        # For features, we want to keep around names like colour = red
                        x_dim, y_dim, z_dim = self.p.shape
                        val_z = np.array(['' for _ in range(z_dim)], dtype=object)
                        val_y = np.array([val_z for _ in range(y_dim)], dtype=object)
                        val = np.array([val_y for _ in range(x_dim)], dtype=object)

                        # Should have same shape
                        assert val.shape == self.p.shape


                        feature_dict = {'bel': np.copy(self.p), "tp" : feature['tp'], "vals": val}
                        self.feature_bels[feature['name']] = feature_dict
                    else:
                        assert False # Shouldn't get here

    
    def pretty_str(self, tp):
        ret_str = "Object: " + str(tp) + "\n\tWidth: " + str(self.map_params['res']) \
                    + " Height: " + str(self.map_params['z_res']) + "\n"
        
        if len(self.feature_bels.keys()) > 0:
            ret_str += "\tFeatures:\n"
            for feature in self.feature_bels.keys():
                ret_str += "\t\t" + str(feature) + "\n"

        return ret_str


    def update(self, obs, eps=1e-6, feature=None, feature_ret_vals=None):
        if feature is None:
            pre_shape = self.p.shape
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

            # Enforce shape
            assert pre_shape == self.p.shape

            print("Done Update")
        else:
            # Update Belief using Binary Bayes Filter
            p = self.feature_bels[feature]['bel']
            log_p = np.where(p > 0, np.log((p+eps)/(1-p+eps)), 0)

            print("p min/max", np.min(p), "/", np.max(p))
            print("obs min/max", np.min(obs), "/", np.max(obs))

            inv_sensor_model = np.where(obs != -1, np.log((obs+eps)/(1-obs+eps)), 0)

            #new_log_p = np.where(inv_sensor_model != 0, log_p + inv_sensor_model, log_p)
            new_log_p = log_p + inv_sensor_model

            new_p = 1 - (1/(1+np.exp(new_log_p)))
            self.feature_bels[feature]['bel'] = new_p

            if np.isnan(new_p).any():
                print("Belief:")
                print(new_p)
                print("Observation:")
                print(obs)
                assert False
            
            del(inv_sensor_model)
            del(new_log_p)
            del(log_p)



            # Update feature vals
            for vox, val in feature_ret_vals:
                val_shape = self.feature_bels[feature]['vals'].shape
                if vox[0] >= val_shape[0] or vox[1] >= val_shape[1] or vox[2] >= val_shape[2]:
                    # Debug shouldn't get here
                    print("Vox: ", vox)
                    print("Vals Shape: ", val_shape)

                self.feature_bels[feature]['vals'][vox[0], vox[1], vox[2]] = val
            
                # Make sure shape is remaining constant
                assert self.feature_bels[feature]['vals'].shape == self.p.shape
            
            # Make sure shape is remaining constant
            assert self.feature_bels[feature]['bel'].shape == self.p.shape

    
    def get_visualization(self):
        return np.mean(self.p, axis=2)
