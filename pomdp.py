from object_belief import *
from reward_func import *
from map_utils import *

class POMDP():
    def __init__(self, query, robot_init_loc, obj_tp_list, trav_map, configs):
        self.prior_beliefs = [] # For AKLD calculation
        self.query = query # Need to keep around for enough info check
        self.loc = robot_init_loc
        self.trav_map = trav_map
        self.configs = configs

        # Calculate originial map params so that we can pass to reward and belief classes
        self.trav_map_original_size = trav_map.trav_map_original_size
        self.trav_map_original_resolution = trav_map.map_default_resolution

        self.bel = {}
        self.reward_funcs = {}
        self.camera_params = configs["camera_params"]
        self.rf_params = configs["rf_params"]
        for obj_tp, num, thresh, relevant_features, priors in obj_tp_list:
            # Get map params based off of current params and obj_tp
            self.map_params = get_map_params(obj_tp, self.trav_map_original_size, self.trav_map_original_resolution)

            if priors == None:
                self.bel[obj_tp] = ObjTpBel(num, threshold, self.map_params, self.configs, relevant_features)
            else:
                self.bel[obj_tp] = priors

            self.reward_funcs[obj_tp] = RewardFunc(map_params, self.camaera_params, self.rf_params)

    def eval_reward(self, obstacle_map, root, node):
        reward = 0
        for obj_tp in self.reward_funcs.keys():
            reward += self.reward_funcs[obj_tp].eval(self.bel[obj_tp].p, obstacle_map, root, node)

    def make_symbolic(self):
        symbolic_info = {}
        for obj_tp in self.bel:
            # Get the locally supressed belief to help localization
            local_bel = non_max_local_supress(self.bel[obj_tp].p)

            local_bel_shape = np.shape(local_bel)

            # TODO: There has to be a better way to do this
            cur_obj_dict = {}
            instance_count = 0
            for x in range(local_bel_shape[0]):
                for y in range(local_bel_shape[1]):
                    for z in range(local_bel_shape[2]):
                        # If passes existence threshold add to objects
                        if local_bel[x, y, z] > self.bel[obj_tp].threshold:
                            # Check features (this is only enum features)
                            feature_dict = {}
                            for feature in self.bel[obj_tp].feature_bels[feature]:
                                if self.bel[obj_tp].feature_bels[feature['name']][x,y,z] > self.bel[obj_tp].feature_bels[feature['name']]['thresh']:
                                    self.bel[obj_tp].feature_bels[feature['name']]['thresh'] < 1:
                                        feature_dict[feature['name']] = 1
                                    else:
                                        feature_dict[feature['name']] = self.bel[obj_tp].feature_bels[feature['name']][x,y,z]

                            feature_dict['location'] = (x,y,z)

                            cur_obj_dict[instance_count] = feature_dict

                            instance_count += 1

            symbolic_info[obj_tp] = cur_obj_dict
        
        return symbolic_info
        

    def enough_info(self):
        self.prior_beliefs.append(self.bel)

        symbolic_info = self.make_symbolic()

        # Pop if over length
        while len(self.prior_beliefs) > self.configs['bel_params']['akld_hist_len']:
            self.prior_beliefs.pop(0)

        # Compute AKLD
        k = len(self.prior_beliefs)
        akld = 0
        for obj_tp in self.bel.keys():
            for i in range(k):
                for j in range(k):
                    akld += kl(self.prior_beliefs[i][obj_tp].p, self.prior_beliefs[j][obj_tp].p)

        akld = akld / (k * (k-1))

        # Check if enough found in symbolic execution
        found_all_obj = True
        num_found = len(self.query.execute(symbolic_info))
        if num_found < self.bel[obj_tp].num:
            found_all_obj = False

        # Check akld and num found
        if akld <= self.configs['bel_params']['akld_stop'] or found_all_obj:
            return True, symbolic_info

        return False, symbolic_info

    def kl(self, p, q):
        return np.sum(np.where(p != 0, p*np.log(p/q), 0))