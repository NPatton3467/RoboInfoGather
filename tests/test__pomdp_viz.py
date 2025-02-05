from RoboInfoGather.dsl import *
from RoboInfoGather.map_utils import *
from RoboInfoGather.program_utils import *

import os
import yaml
import random

from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image


# Load the config
config_filename = os.path.join(f"/robodata/user_data/npatt/explore-eqa/RoboInfoGather/info_gather.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

# Get trav_map
og_resolution = config['scene']['trav_map_resolution']
trav_map = get_trav_map(config['scene']['trav_map_path'], config['scene']['floor'], og_resolution, og_resolution)
map_size, _ = trav_map.shape

# Get Position/orientation
pos = np.array([4.31607878e-05, -6.68951543e-08,  9.91712511e-03])
ori = np.array([4.32668778e-04, -3.11666692e-04, -3.01003456e-06,  9.99999881e-01])

# Avg Cup Volume where red + average bowl volume where blue
print("Avg Cup Vol Where red + avg bowl volume where blue")
new_where_cup = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
new_query_cup = Query(obj_tp="cup", where_clause=new_where_cup)
print(new_query_cup.pretty_str())
new_map_cup = Map(obj_tp='cup', map_feature='volume', query=new_query_cup, map_tp='feature_scalar')
print(new_map_cup.pretty_str())
new_agg_cup = Aggregator(agg_tp='avg', symbolic_list=new_map_cup)
print(new_agg_cup.pretty_str())

new_where_bowl = WhereClause(where_tp='feature_enum', obj_tp="bowl", enum_feature='colour', enum_param='blue')
new_query_bowl = Query(obj_tp="bowl", where_clause=new_where_bowl)
print(new_query_bowl.pretty_str())
new_map_bowl = Map(obj_tp='bowl', map_feature='volume', query=new_query_bowl, map_tp='feature_scalar')
print(new_map_bowl.pretty_str())
new_agg_bowl = Aggregator(agg_tp='avg', symbolic_list=new_map_bowl)
print(new_agg_bowl.pretty_str())

new_prim = Primitives(prim_tp='op', prim=new_agg_cup, prim2=new_agg_bowl, prim_op='plus')
print(new_prim.pretty_str())

new_pomdp = gen_pomdp_from_query(query=new_prim, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

print("POMDP Belief: ", new_pomdp.bel)
print("cup features: ", new_pomdp.bel['cup'].feature_bels)
print("cup map params: ", new_pomdp.bel['cup'].map_params)
print("bowl features: ", new_pomdp.bel['bowl'].feature_bels)
print("bowl map params: ", new_pomdp.bel['bowl'].map_params)

print("POMDP Reward: ", new_pomdp.reward_funcs)
print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)
print("Reward Func Map Params: ", new_pomdp.reward_funcs['bowl'].map_params)

new_pomdp.visualize()

# Randomly update points in belief to 1 and make sure viz changes
i = 0
while i < 10000:
    i += 1

    rand_x_cup = random.randint(0, np.shape(new_pomdp.bel['cup'].p)[0]-1)
    rand_y_cup = random.randint(0, np.shape(new_pomdp.bel['cup'].p)[1]-1)

    rand_x_bowl = random.randint(0, np.shape(new_pomdp.bel['bowl'].p)[0]-1)
    rand_y_bowl = random.randint(0, np.shape(new_pomdp.bel['bowl'].p)[1]-1)

    new_pomdp.bel['cup'].p[rand_x_cup, rand_y_cup] = 1
    new_pomdp.bel['bowl'].p[rand_x_bowl, rand_y_bowl] = 1

    new_pomdp.visualize()
