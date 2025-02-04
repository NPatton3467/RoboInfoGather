from RoboInfoGather.object_belief import *
from RoboInfoGather.reward_func import *
from RoboInfoGather.map_utils import *

from matplotlib import pyplot as plt

import copy
import math
import torch

class POMDP():
    def __init__(self, query, robot_init_loc, obj_tp_list, trav_map_og_size, trav_map_og_res, configs):
        # Stopping criteria type
        self.explore_stop = configs['bel_params']['explore_stop']

        self.prior_beliefs = [] # For AKLD calculation
        self.prior_beliefs_quick = {}
        self.query = query # Need to keep around for enough info check
        self.loc = robot_init_loc
        self.configs = configs

        # Calculate originial map params so that we can pass to reward and belief classes
        self.trav_map_original_size = trav_map_og_size
        self.trav_map_original_resolution = trav_map_og_res

        self.bel = {}
        self.reward_funcs = {}
        self.camera_params = configs["camera_params"]
        self.rf_params = configs["rf_params"]
        for obj_tp, num, thresh, relevant_features, priors in obj_tp_list:
            # Get map params based off of current params and obj_tp
            map_params = get_map_params(obj_tp, self.trav_map_original_size, self.trav_map_original_resolution)

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
            for x in range(local_bel_shape[0]):
                for y in range(local_bel_shape[1]):
                    for z in range(local_bel_shape[2]):
                        # If passes existence threshold add to objects
                        if local_bel.p[x, y, z] > local_bel.threshold:
                            # Check features (this is only enum features)
                            feature_dict = {}
                            for feature in local_bel.feature_bels:
                                feature_dict[feature] = local_bel.feature_bels[feature]['vals'][x,y,z]                                
                            
                            map_resolution = local_bel.map_params['res']
                            map_size = local_bel.map_params['size']
                            w_xy = map_to_world(np.array([x,y]), map_resolution, map_size)
                            w_z = z * local_bel.map_params['z_res']
                            feature_dict['location'] = (w_xy[0],w_xy[1],w_z)

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

            cur_p_ent = np.where((p > 0) & (p < 1), p * np.log(p) + (1-p)*np.log(1-p), 0)
            print(np.min(cur_p_ent))
            num_valid = np.sum(np.where((p > 0) & (p < 1), 1, 0))
            cur_p_avg_ent = np.sum(cur_p_ent) / num_valid

            avg_entropy += (cur_p_avg_ent / len(self.bel.keys()))

        return -1 * avg_entropy

    def enough_info(self, iteration):
        symbolic_info = self.make_symbolic()
        
        # Check if enough found in symbolic execution
        found_all_obj = True
        current_symbolic_query = self.query.execute(symbolic_info)
        for obj_tp in self.bel.keys():
            if obj_tp not in current_symbolic_query:
                found_all_obj = False
                break

            num_found = len(current_symbolic_query[obj_tp])
            if num_found < self.bel[obj_tp].num or self.bel[obj_tp].num == -1:
                found_all_obj = False
                break
        
        if self.explore_stop == "AKLD":
            akld = self.compute_akld(iterations)
            # Check akld and num found
            if (akld > 0 and akld <= self.configs['bel_params']['akld_stop']) or found_all_obj:
                print("Done POMDP Execution -- AKLD: ", akld, " Num Found: ", num_found, " Required Number to Find: ", self.bel[obj_tp].num, " found_all_obj: ", found_all_obj)
                return True, symbolic_info

        elif self.explore_stop == "ENTROPY":
            avg_ent = self.compute_average_entropy()

            if avg_ent <= self.configs['bel_params']['avg_ent_stop'] or found_all_obj:
                print("Done POMDP Execution -- ENTROPY: ", avg_ent, " Num Found: ", num_found, " Required Number to Find: ", self.bel[obj_tp].num, " found_all_obj: ", found_all_obj)
                return True, symbolic_info


        return False, symbolic_info

    def kl(self, p, q):
        # KL is sum over all possibilities of rv X
        # Here obj either exists or doesn't so p(x = 1) = 1 - p(x=0)
        kl1 = np.sum(np.where((p > 0) & (q > 0), p*np.log(p/q), 0))
        kl0 = np.sum(np.where(((1-p) > 0) & ((1-q) > 0), (1-p)*np.log((1-p)/(1-q)), 0))

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
