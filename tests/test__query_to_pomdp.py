from RoboInfoGather.dsl import *
from RoboInfoGather.map_utils import *
from RoboInfoGather.program_utils import *

import os
import yaml

# import omnigibson as og
# from omnigibson.utils.ui_utils import choose_from_options

from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image

# from omnigibson.object_states.pose import Pose

# Load the config
config_filename = os.path.join(f"./RoboInfoGather/info_gather.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

# # Start omni gibson
# # check if we want to quick load or full load the scene
# load_options = {
#     "Quick": "Only load the building assets (i.e.: the floors, walls, doors)",
#     "Full": "Load all interactive objects in the scene",
# }
# load_mode = choose_from_options(options=load_options, name="load mode", random_selection=False)
# if load_mode == "Quick":
#     config["scene"]["load_object_categories"] = ["floors", "walls", "door"]

# # Load the environment
# env = og.Environment(configs=config)

# # Allow user to move camera more easily
# og.sim.enable_viewer_camera_teleoperation()

# # Reset env before start? 
# og.log.info("Resetting environment")
# env.reset()

# Get trav_map
# og_resolution = env.scene._trav_map.map_default_resolution
og_resolution = config['scene']['trav_map_resolution']
trav_map = get_trav_map(config['scene']['trav_map_path'], config['scene']['floor'], og_resolution, og_resolution)
map_size, _ = trav_map.shape

# Get Position/orientation
pos = np.array([4.31607878e-05, -6.68951543e-08,  9.91712511e-03])
ori = np.array([4.32668778e-04, -3.11666692e-04, -3.01003456e-06,  9.99999881e-01])
# pos, ori = env.robots[0].get_position_orientation()

# True WhereClause single obj
# print("True where clause -- single object")
# new_where = WhereClause(where_tp='true', obj_tp="cup")
# print(new_where.pretty_str())

# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# print("POMDP Belief: ", new_pomdp.bel)
# print("cup features: ", new_pomdp.bel['cup'].feature_bels)
# print("cup map params: ", new_pomdp.bel['cup'].map_params)

# print("POMDP Reward: ", new_pomdp.reward_funcs)
# print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)
# print("\n\n")


# # Enum WhereClause single obj
# print("Where colour = red -- single object")
# new_where = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
# print(new_where.pretty_str())

# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# print("POMDP Belief: ", new_pomdp.bel)
# print("cup features: ", new_pomdp.bel['cup'].feature_bels)
# print("cup map params: ", new_pomdp.bel['cup'].map_params)

# print("POMDP Reward: ", new_pomdp.reward_funcs)
# print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)


# # And where clause single obj
# print("Where colour = red and volume > 300 ml")
# sub_where_colour = WhereClause(where_tp='feature_enum', obj_tp='cup', enum_feature='colour', enum_param='red')
# print(sub_where_colour.pretty_str())

# sub_where_volume = WhereClause(where_tp='feature_scalar', obj_tp='cup', scalar_feature='volume', scalar_param='300', scalar_comparator='Gt')
# print(sub_where_volume.pretty_str())

# new_where = WhereClause(where_tp='and', obj_tp='cup', sub_where_clause=[sub_where_colour, sub_where_volume])
# print(new_where.pretty_str())

# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# print("POMDP Belief: ", new_pomdp.bel)
# print("cup features: ", new_pomdp.bel['cup'].feature_bels)
# print("cup map params: ", new_pomdp.bel['cup'].map_params)

# print("POMDP Reward: ", new_pomdp.reward_funcs)
# print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)

# # Nested where clause single obj
# print("Where colour = red and volume > 300 ml and height < 0.2m")
# sub_where_colour = WhereClause(where_tp='feature_enum', obj_tp='cup', enum_feature='colour', enum_param='red')
# print(sub_where_colour.pretty_str())

# sub_where_volume = WhereClause(where_tp='feature_scalar', obj_tp='cup', scalar_feature='volume', scalar_param='300', scalar_comparator='Gt')
# print(sub_where_volume.pretty_str())

# sub_where_height = WhereClause(where_tp='feature_scalar', obj_tp='cup', scalar_feature='height', scalar_param='0.2', scalar_comparator='Lt')
# print(sub_where_height.pretty_str())

# sub_where_first_and = WhereClause(where_tp='and', obj_tp='cup', sub_where_clause=[sub_where_colour, sub_where_volume])

# new_where = WhereClause(where_tp='and', obj_tp='cup', sub_where_clause=[sub_where_first_and, sub_where_height])
# print(new_where.pretty_str())

# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# print("POMDP Belief: ", new_pomdp.bel)
# print("cup features: ", new_pomdp.bel['cup'].feature_bels)
# print("cup map params: ", new_pomdp.bel['cup'].map_params)

# print("POMDP Reward: ", new_pomdp.reward_funcs)
# print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)

# # Max where clause single obj
# print("Where max(volume)")
# new_where = WhereClause(where_tp='max', obj_tp='cup', scalar_feature='volume')
# print(new_where.pretty_str())

# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# print("POMDP Belief: ", new_pomdp.bel)
# print("cup features: ", new_pomdp.bel['cup'].feature_bels)
# print("cup map params: ", new_pomdp.bel['cup'].map_params)

# print("POMDP Reward: ", new_pomdp.reward_funcs)
# print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)

# # Map Volume single obj
# print("Cup Volume")
# new_where = WhereClause(where_tp='true', obj_tp="cup")
# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_map = Map(obj_tp='cup', map_feature='volume', query=new_query)
# print(new_map.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_map, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# print("POMDP Belief: ", new_pomdp.bel)
# print("cup features: ", new_pomdp.bel['cup'].feature_bels)
# print("cup map params: ", new_pomdp.bel['cup'].map_params)

# print("POMDP Reward: ", new_pomdp.reward_funcs)
# print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)

# # Map Volume single obj where red
# print("Cup Volume Where red")
# new_where = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_map = Map(obj_tp='cup', map_feature='volume', query=new_query, map_tp='feature_scalar')
# print(new_map.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_map, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# print("POMDP Belief: ", new_pomdp.bel)
# print("cup features: ", new_pomdp.bel['cup'].feature_bels)
# print("cup map params: ", new_pomdp.bel['cup'].map_params)

# print("POMDP Reward: ", new_pomdp.reward_funcs)
# print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)

# # Get Nth Volume single obj where red
# print("Cup Volume Where red")
# new_where = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_map = Map(obj_tp='cup', map_feature='volume', query=new_query, map_tp='feature_scalar')
# print(new_map.pretty_str())

# new_getnth = GetNth(new_map, 2)
# print(new_getnth.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_getnth, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# print("POMDP Belief: ", new_pomdp.bel)
# print("cup features: ", new_pomdp.bel['cup'].feature_bels)
# print("cup map params: ", new_pomdp.bel['cup'].map_params)

# print("POMDP Reward: ", new_pomdp.reward_funcs)
# print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)

# Count single obj where red
# print("Count Cup  Where red")
# new_where = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_count = Count(new_query, 'cup')
# print(new_count.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_count, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# print("POMDP Belief: ", new_pomdp.bel)
# print("cup features: ", new_pomdp.bel['cup'].feature_bels)
# print("cup map params: ", new_pomdp.bel['cup'].map_params)

# print("POMDP Reward: ", new_pomdp.reward_funcs)
# print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)

# # Avg Cup Volume single obj where red
# print("Avg Cup Vol Where red")
# new_where = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_map = Map(obj_tp='cup', map_feature='volume', query=new_query, map_tp='feature_scalar')
# print(new_map.pretty_str())

# new_agg = Aggregator(agg_tp='avg', symbolic_list=new_map)
# print(new_agg.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_agg, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# print("POMDP Belief: ", new_pomdp.bel)
# print("cup features: ", new_pomdp.bel['cup'].feature_bels)
# print("cup map params: ", new_pomdp.bel['cup'].map_params)

# print("POMDP Reward: ", new_pomdp.reward_funcs)
# print("Reward Func Map Params: ", new_pomdp.reward_funcs['cup'].map_params)

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