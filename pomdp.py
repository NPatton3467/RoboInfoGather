from RoboInfoGather.object_belief import *
from RoboInfoGather.reward_func import *
from RoboInfoGather.map_utils import *
from RoboInfoGather.determinization_utils import *
from RoboInfoGather.observation_utils import *

from matplotlib import pyplot as plt

import copy
import math
import torch
import numpy as np

class POMDP():
    def __init__(self, query, robot_init_loc, obj_tp_list, trav_map_og_dim, trav_map_og_res, vol_origin, configs):
        # Stopping criteria type
        self.explore_stop = configs['bel_params']['explore_stop']

        self.prior_beliefs = [] # For AKLD calculation
        self.prior_beliefs_quick = {}
        self.query = query # Need to keep around for enough info check
        self.loc = robot_init_loc
        self.configs = configs

        # Calculate originial map params so that we can pass to reward and belief classes
        self.trav_map_original_dim = trav_map_og_dim
        self.trav_map_original_resolution = trav_map_og_res
        self.vol_origin = vol_origin

        self.bel = {}
        self.reward_funcs = {}
        self.camera_params = configs["camera_params"]
        self.rf_params = configs["rf_params"]
        for obj_tp, num, thresh, relevant_features, priors in obj_tp_list:
            # Get map params based off of current params and obj_tp
            map_params = get_map_params(obj_tp, self.trav_map_original_dim, self.trav_map_original_resolution, self.vol_origin, configs)

            if priors == None:
                self.bel[obj_tp] = ObjTpBel(num, thresh, map_params, self.configs, relevant_features)
            else:
                self.bel[obj_tp] = priors

            self.reward_funcs[obj_tp] = RewardFunc(map_params, self.camera_params, self.rf_params)

        # Save figures for drawing (one per belief)
        if self.configs['bel_params']['visualize']:
            self.figures = {}
            for obj_tp in self.bel:
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                plt.ion()
                plt.show()
                self.figures[obj_tp] = (fig, ax)


    def pretty_str(self):
        ret_str = ""
        for obj_tp in self.bel.keys():
            ret_str += self.bel[obj_tp].pretty_str(obj_tp)

        return ret_str


    def eval_reward(self, obstacle_map, root, node):
        reward = 0
        for obj_tp in self.reward_funcs.keys():
            reward += self.reward_funcs[obj_tp].eval(self.bel[obj_tp].p, obstacle_map, root, node)

    def make_symbolic(self):
        symbolic_info = {}
        for obj_tp in self.bel:
            # Get the locally supressed belief to help localization
            local_bel = self.bel[obj_tp]

            local_bel_shape = local_bel.p.shape

            # TODO: There has to be a better way to do this
            cur_obj_dict = {}
            instance_count = 0
            xyzs = suppress_non_max(local_bel)
            for (xt, yt, zt) in xyzs:
                # Check features (this is only enum features)
                x = int(xt.detach().cpu())
                y = int(yt.detach().cpu())
                z = int(zt.detach().cpu())
                print(x, y, z)
                feature_dict = {}
                for feature in local_bel.feature_bels:
                    feature_val = local_bel.feature_bels[feature]['vals'][x,y,z]
                    if type(feature_val) is torch.Tensor:
                        feature_val = float(feature_val.detach().cpu())

                    feature_dict[feature] = feature_val
                
                map_resolution = local_bel.map_params['res']
                z_resolution = local_bel.map_params['z_res']
                vol_origin = local_bel.map_params['vol_origin']
                
                w_xyz = map_to_world(np.array([x,y,z]), vol_origin, map_resolution, z_resolution)
                feature_dict['location'] = (w_xyz[0],w_xyz[1],w_xyz[2])

                cur_obj_dict[instance_count] = feature_dict

                instance_count += 1

            symbolic_info[obj_tp] = cur_obj_dict
        
        return symbolic_info
        

    def compute_akld(self, iterations):
        if iteration % self.configs['bel_params']['akld_append_interval'] == 0:
            self.prior_beliefs.append(copy.deepcopy(self.bel))

        # Pop if over length
        while len(self.prior_beliefs) > self.configs['bel_params']['akld_hist_len']:
            self.prior_beliefs.pop(0)

        # Compute AKLD
        k = len(self.prior_beliefs)
        akld = 0
        for obj_tp in self.bel.keys():
            for i in range(k):
                for j in range(i+1, k):
                    akld += self.kl(self.prior_beliefs[i][obj_tp].p, self.prior_beliefs[j][obj_tp].p)


        if k == self.configs['bel_params']['akld_hist_len']:
            akld = akld / (k * (k-1))
            print("AKLD: ", akld)
        else:
            akld = -1

        return akld

    def compute_average_entropy(self):
        avg_entropy = 0.0

        for key in self.bel.keys():
            p = self.bel[key].p

            cur_p_ent = torch.where((p > 0) & (p < 1), p * torch.log(p) + (1-p)*torch.log(1-p), 0)
            print(torch.min(cur_p_ent))
            num_valid = torch.sum(torch.where((p > 0) & (p < 1), 1, 0))
            cur_p_avg_ent = torch.sum(cur_p_ent) / num_valid

            avg_entropy += (cur_p_avg_ent.detach().cpu() / len(self.bel.keys()))

        return -1 * avg_entropy

    def enough_info(self, iteration):
        symbolic_info = self.make_symbolic()
        
        # Check if enough found in symbolic execution
        found_all_obj = True
        current_symbolic_query = self.query.execute(symbolic_info)
        total_num_found = 0
        for obj_tp in self.bel.keys():
            if obj_tp in current_symbolic_query:
                num_found = len(current_symbolic_query[obj_tp])
                total_num_found+=num_found
            if obj_tp not in current_symbolic_query:
                found_all_obj = False
                break

            if num_found < self.bel[obj_tp].num or self.bel[obj_tp].num == -1:
                found_all_obj = False
                break
        
        if self.explore_stop == "AKLD":
            akld = self.compute_akld(iterations)
            # Check akld and num found
            if (akld > 0 and akld <= self.configs['bel_params']['akld_stop']) or found_all_obj:
                print("Done POMDP Execution -- AKLD: ", akld, " Num Found: ", total_num_found, " Required Number to Find: ", self.bel[obj_tp].num, " found_all_obj: ", found_all_obj)
                return True, symbolic_info

        elif self.explore_stop == "ENTROPY":
            avg_ent = self.compute_average_entropy()

            if avg_ent <= self.configs['bel_params']['avg_ent_stop'] or found_all_obj:
                print("Done POMDP Execution -- ENTROPY: ", avg_ent, " Num Found: ", total_num_found, " Required Number to Find: ", self.bel[obj_tp].num, " found_all_obj: ", found_all_obj)
                return True, symbolic_info


        return False, symbolic_info

    def kl(self, p, q):
        # KL is sum over all possibilities of rv X
        # Here obj either exists or doesn't so p(x = 1) = 1 - p(x=0)
        kl1 = torch.sum(torch.where((p > 0) & (q > 0), p*torch.log(p/q), 0))
        kl0 = torch.sum(torch.where(((1-p) > 0) & ((1-q) > 0), (1-p)*torch.log((1-p)/(1-q)), 0))

        kl = kl0+kl1

        #print("KL0: ", kl0, " KL1: ", kl1, " KL: ", kl)
        #print("Equal?: ", (p==q).all())

        return kl

    def visualize(self, rob_pos):
        for obj_tp in self.figures:
            visualization = self.bel[obj_tp].get_visualization()

            self.figures[obj_tp][1].imshow(visualization, interpolation='nearest')

        plt.draw()
        plt.pause(0.001)

        print(self.figures)

    def update(
            self,
            vlm,
            molmo_tools,
            angle,
            camera_pos,
            cam_pose_normal,
            rgb, 
            depth,
            dino_model,
            RIG_config,
            tsdf_planner,
            cam_intr,
            cnt_step,
            pts,
            debug_f_path
        ):

        found_obj = False
        ret_pix_coords = []
        ret_real_coords = []
        for obj_tp in self.bel.keys():
            print(self.bel[obj_tp])
            # Get predictions for all voxels based on observations
            vox_preds, _, found_obj, pix_coords, real_world_coords = get_vox_preds(
                    vlm,
                    molmo_tools,
                    angle,
                    camera_pos,
                    cam_pose_normal,
                    self.bel[obj_tp],
                    obj_tp,
                    rgb,
                    depth,
                    dino_model,
                    RIG_config,
                    tsdf_planner,
                    cam_intr,
                    iteration=cnt_step
                )

            self.bel[obj_tp].update(vox_preds)
            print("Camera Pose: ", cam_pose_normal)
            print("PTS: ", pts)

            ret_pix_coords += pix_coords
            ret_real_coords += real_world_coords

            np.save(debug_f_path+f"bel_{obj_tp}_{cnt_step}.npy", np.array(self.bel[obj_tp].p.detach().cpu()))

            # Do the same for each feature
            print("Starting Feature Update in run_RIG")
            for feature in self.bel[obj_tp].feature_bels.keys():
                # Get predictions for all voxels based on observations
                print(self.bel[obj_tp])
                vox_preds, feature_ret_vals, found_obj, pix_coords, real_world_coords = get_vox_preds(
                        vlm,
                        molmo_tools,
                        angle,
                        camera_pos,
                        cam_pose_normal,
                        self.bel[obj_tp],
                        obj_tp,
                        rgb,
                        depth,
                        dino_model,
                        RIG_config,
                        tsdf_planner,
                        cam_intr,
                        feature=feature,
                        iteration=cnt_step
                    )

                self.bel[obj_tp].update(vox_preds, feature=feature, feature_ret_vals=feature_ret_vals)

        print("Done Feature Update in run_RIG")
        return ret_real_coords, ret_pix_coords, found_obj
