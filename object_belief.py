import json
import torch
import numpy as np

from RoboInfoGather.map_utils import *

class ObjTpBel():
    def __init__(self, num, threshold, map_params, configs, relevant_features=None):
        self.num = num # Num objects to be found, If None -> unbounded
        self.threshold = 0.75 # threshold # Existence threshold
        self.map_params = map_params


        self.configs = configs
        self.relevant_features = relevant_features
        
        print("About to make first tensor")
        dim = self.map_params['dim']
        dim_tup = (dim[0], dim[1], dim[2])
        self.device = self.configs['bel_params']['torch_device']
        self.p = torch.ones(dim_tup).to(device=torch.device(self.device))
        self.p = self.p * 0.5

        print("Made first tensor: ", self.p.shape)

        # Belief over features
        self.feature_bels = {}
        if self.relevant_features != None and self.relevant_features != [None]:
            for feature in self.relevant_features:
                if feature['name'] != None:
                    if feature['tp'] == "feature_scalar":
                        feature_dict = {'bel': torch.clone(self.p).to(torch.device(self.device)), "tp" : feature['tp'], "vals": torch.zeros_like(self.p).to(torch.device(self.device))}
                        self.feature_bels[feature['name']] = feature_dict
                    elif feature['tp'] == "feature_enum":
                        # For features, we want to keep around names like colour = red
                        x_dim, y_dim, z_dim = self.p.shape
                        val_z = np.array(['' for _ in range(z_dim)], dtype=object)
                        val_y = np.array([val_z for _ in range(y_dim)], dtype=object)
                        val = np.array([val_y for _ in range(x_dim)], dtype=object)

                        # Should have same shape
                        assert val.shape == self.p.shape


                        feature_dict = {'bel': torch.clone(self.p).to(torch.device(self.device)), "tp" : feature['tp'], "vals": val}
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

        """
        Update the object belief with the observation (voxel values already computed). 
        If feature is not none, store the feature values (feature_ret_vals) at the corresponding
        voxel, and update the probability of being observed correct (obs: voxel values already
        computed).

        Inputs:
            obs:                A set of voxels (same size and shape as belief) where the value
                                    [0,1] predicts the probability of the belief value being correct from the 
                                    most recent observation.

                                    If the belief being updated is not a feature belief, it's the probability
                                    of an instance of the object type existing at that voxel, from the most
                                    recent observation.
            feature:            If not none, this string is used to select which feature we are updating
                                    the predictions for
            feature_ret_vals:   A list of pairs of (voxel-coordinates, value) which is used to set the most
                                    likely value of 'feature' at the given voxel
        """

        if feature is None:
            pre_shape = self.p.shape
            # Update Belief using Binary Bayes Filter
            log_p = torch.where(self.p > 0, torch.log((self.p+eps)/(1-self.p+eps)), 0)

            print("p min/max", torch.min(self.p), "/", torch.max(self.p))
            print("obs min/max", torch.min(obs), "/", torch.max(obs))

            inv_sensor_model = torch.where(obs != -1, torch.log((obs+eps)/(1-obs+eps)), 0)

            new_log_p = log_p + inv_sensor_model

            self.p = 1 - (1/(1+torch.exp(new_log_p)))

            if torch.isnan(self.p).any():
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
            log_p = torch.where(p > 0, torch.log((p+eps)/(1-p+eps)), 0)

            print("p min/max", torch.min(p), "/", torch.max(p))
            print("obs min/max", torch.min(obs), "/", torch.max(obs))

            inv_sensor_model = torch.where(obs != -1, torch.log((obs+eps)/(1-obs+eps)), 0)

            new_log_p = log_p + inv_sensor_model

            new_p = 1 - (1/(1+torch.exp(new_log_p)))
            self.feature_bels[feature]['bel'] = new_p

            if torch.isnan(new_p).any():
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

                try:
                    self.feature_bels[feature]['vals'][vox[0], vox[1], vox[2]] = val
                except Exception as e:
                    print(f"Failed to make val update into belief: {str(e)}")
            
                # Make sure shape is remaining constant
                assert self.feature_bels[feature]['vals'].shape == self.p.shape
            
            # Make sure shape is remaining constant
            assert self.feature_bels[feature]['bel'].shape == self.p.shape

    
    def get_visualization(self):
        return torch.mean(self.p, axis=2)
