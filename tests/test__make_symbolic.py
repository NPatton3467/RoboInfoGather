from RoboInfoGather.dsl import *
from RoboInfoGather.map_utils import *
from RoboInfoGather.program_utils import *

import os
import yaml

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


# # One cup, no whereclause = True
# print("One cup, whereclause = true")
# new_where = WhereClause(where_tp='true', obj_tp="cup")

# new_query = Query(obj_tp="cup", where_clause=new_where)

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# new_pomdp.bel['cup'].p[10,10,10] = 0.95

# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")

# # 5 cups, no whereclause = True
# print("Five cup, whereclause = true")
# new_where = WhereClause(where_tp='true', obj_tp="cup")

# new_query = Query(obj_tp="cup", where_clause=new_where)

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# new_pomdp.bel['cup'].p[10,10,10] = 0.95
# new_pomdp.bel['cup'].p[15,10,10] = 0.95
# new_pomdp.bel['cup'].p[10,15,10] = 0.95
# new_pomdp.bel['cup'].p[10,10,15] = 0.95
# new_pomdp.bel['cup'].p[0,10,10] = 0.95

# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")


# # Enum WhereClause single obj
# print("Where colour = red -- single object")
# new_where = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
# print(new_where.pretty_str())

# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# new_pomdp.bel['cup'].p[10,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 'red'

# new_pomdp.bel['cup'].p[15,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 'black'

# new_pomdp.bel['cup'].p[10,15,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 'green'

# new_pomdp.bel['cup'].p[10,10,15] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 'yellow'

# new_pomdp.bel['cup'].p[0,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 'red'

# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")


# # And where clause single obj
# print("Where colour = red and volume > 300 ml")
# sub_where_colour = WhereClause(where_tp='feature_enum', obj_tp='cup', enum_feature='colour', enum_param='red')

# sub_where_volume = WhereClause(where_tp='feature_scalar', obj_tp='cup', scalar_feature='volume', scalar_param='300', scalar_comparator='Gt')

# new_where = WhereClause(where_tp='and', obj_tp='cup', sub_where_clause=[sub_where_colour, sub_where_volume])

# new_query = Query(obj_tp="cup", where_clause=new_where)

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)


# new_pomdp.bel['cup'].p[10,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,10] = 0.295 # Volume in liters 

# new_pomdp.bel['cup'].p[15,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 'black'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][15,10,10] = 0.310 # Volume in liters 

# new_pomdp.bel['cup'].p[10,15,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 'green'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,15,10] = 0.400 # Volume in liters 

# new_pomdp.bel['cup'].p[10,10,15] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 'yellow'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,15] = 0.250 # Volume in liters 

# new_pomdp.bel['cup'].p[0,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][0,10,10] = 0.350 # Volume in liters 


# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")

# # Nested where clause single obj
# print("Where colour = red and volume > 300 ml and height < 0.2m")
# sub_where_colour = WhereClause(where_tp='feature_enum', obj_tp='cup', enum_feature='colour', enum_param='red')

# sub_where_volume = WhereClause(where_tp='feature_scalar', obj_tp='cup', scalar_feature='volume', scalar_param='300', scalar_comparator='Gt')

# sub_where_height = WhereClause(where_tp='feature_scalar', obj_tp='cup', scalar_feature='height', scalar_param='0.2', scalar_comparator='Lt')

# sub_where_first_and = WhereClause(where_tp='and', obj_tp='cup', sub_where_clause=[sub_where_colour, sub_where_volume])

# new_where = WhereClause(where_tp='and', obj_tp='cup', sub_where_clause=[sub_where_first_and, sub_where_height])

# new_query = Query(obj_tp="cup", where_clause=new_where)

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# new_pomdp.bel['cup'].p[10,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,10] = 0.295 # Volume in liters 
# new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,10] = 0.3 # Height in m

# new_pomdp.bel['cup'].p[15,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 'black'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][15,10,10] = 0.310 # Volume in liters 
# new_pomdp.bel['cup'].feature_bels['height']['bel'][15,10,10] = 0.31 # Height in m

# new_pomdp.bel['cup'].p[10,15,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 'green'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,15,10] = 0.400 # Volume in liters 
# new_pomdp.bel['cup'].feature_bels['height']['bel'][10,15,10] = 0.32 # Height in m

# new_pomdp.bel['cup'].p[10,10,15] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 'yellow'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,15] = 0.250 # Volume in liters 
# new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,15] = 0.33 # Height in m

# new_pomdp.bel['cup'].p[0,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][0,10,10] = 0.350 # Volume in liters 
# new_pomdp.bel['cup'].feature_bels['height']['bel'][0,10,10] = 0.19 # Height in m


# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")


# # Max where clause single obj
# print("Where max(volume)")
# new_where = WhereClause(where_tp='max', obj_tp='cup', scalar_feature='volume')
# print(new_where.pretty_str())

# new_query = Query(obj_tp="cup", where_clause=new_where)
# print(new_query.pretty_str())

# new_pomdp = gen_pomdp_from_query(query=new_query, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# new_pomdp.bel['cup'].p[10,10,10] = 0.95
# #new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,10] = 0.295 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,10] = 0.3 # Height in m

# new_pomdp.bel['cup'].p[15,10,10] = 0.95
# #new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 'black'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][15,10,10] = 0.310 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][15,10,10] = 0.31 # Height in m

# new_pomdp.bel['cup'].p[10,15,10] = 0.95
# #new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 'green'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,15,10] = 0.400 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,15,10] = 0.32 # Height in m

# new_pomdp.bel['cup'].p[10,10,15] = 0.95
# #new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 'yellow'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,15] = 0.250 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,15] = 0.33 # Height in m

# new_pomdp.bel['cup'].p[0,10,10] = 0.95
# #new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][0,10,10] = 0.350 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][0,10,10] = 0.19 # Height in m


# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")

# # Map Volume single obj
# print("Cup Volume")
# new_where = WhereClause(where_tp='true', obj_tp="cup")
# new_query = Query(obj_tp="cup", where_clause=new_where)

# new_map = Map(obj_tp='cup', map_feature='volume', query=new_query)

# new_pomdp = gen_pomdp_from_query(query=new_map, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)


# new_pomdp.bel['cup'].p[10,10,10] = 0.95
# #new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,10] = 0.295 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,10] = 0.3 # Height in m

# new_pomdp.bel['cup'].p[15,10,10] = 0.95
# #new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 'black'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][15,10,10] = 0.310 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][15,10,10] = 0.31 # Height in m

# new_pomdp.bel['cup'].p[10,15,10] = 0.95
# #new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 'green'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,15,10] = 0.400 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,15,10] = 0.32 # Height in m

# new_pomdp.bel['cup'].p[10,10,15] = 0.95
# #new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 'yellow'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,15] = 0.250 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,15] = 0.33 # Height in m

# new_pomdp.bel['cup'].p[0,10,10] = 0.95
# #new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][0,10,10] = 0.350 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][0,10,10] = 0.19 # Height in m


# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")


# # Map Volume single obj where red
# print("Cup Volume Where red")
# new_where = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
# new_query = Query(obj_tp="cup", where_clause=new_where)

# new_map = Map(obj_tp='cup', map_feature='volume', query=new_query, map_tp='feature_scalar')

# new_pomdp = gen_pomdp_from_query(query=new_map, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)



# new_pomdp.bel['cup'].p[10,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,10] = 0.295 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,10] = 0.3 # Height in m

# new_pomdp.bel['cup'].p[15,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 'black'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][15,10,10] = 0.310 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][15,10,10] = 0.31 # Height in m

# new_pomdp.bel['cup'].p[10,15,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 'green'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,15,10] = 0.400 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,15,10] = 0.32 # Height in m

# new_pomdp.bel['cup'].p[10,10,15] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 'yellow'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,15] = 0.250 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,15] = 0.33 # Height in m

# new_pomdp.bel['cup'].p[0,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][0,10,10] = 0.350 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][0,10,10] = 0.19 # Height in m


# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")

# # Get Nth Volume single obj where red
# print("Cup Volume Where red")
# new_where = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
# new_query = Query(obj_tp="cup", where_clause=new_where)

# new_map = Map(obj_tp='cup', map_feature='volume', query=new_query, map_tp='feature_scalar')

# new_getnth = GetNth(new_map, 2)

# new_pomdp = gen_pomdp_from_query(query=new_getnth, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# new_pomdp.bel['cup'].p[10,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,10] = 0.295 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,10] = 0.3 # Height in m

# new_pomdp.bel['cup'].p[15,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 'black'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][15,10,10] = 0.310 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][15,10,10] = 0.31 # Height in m

# new_pomdp.bel['cup'].p[10,15,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 'green'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,15,10] = 0.400 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,15,10] = 0.32 # Height in m

# new_pomdp.bel['cup'].p[10,10,15] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 'yellow'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,15] = 0.250 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,15] = 0.33 # Height in m

# new_pomdp.bel['cup'].p[0,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][0,10,10] = 0.350 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][0,10,10] = 0.19 # Height in m


# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")

# assert symbolic_info == {'cup': {0: {'volume': 0.35, 'colour': 'red', 'location': (0, 10, 10)}, 1: {'volume': 0.295, 'colour': 'red', 'location': (10, 10, 10)}, 2: {'volume': 0.25, 'colour': 'yellow', 'location': (10, 10, 15)}, 3: {'volume': 0.4, 'colour': 'green', 'location': (10, 15, 10)}, 4: {'volume': 0.31, 'colour': 'black', 'location': (15, 10, 10)}}}

# # Count single obj where red
# print("Count Cup  Where red")
# new_where = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
# new_query = Query(obj_tp="cup", where_clause=new_where)

# new_count = Count(new_query, 'cup')

# new_pomdp = gen_pomdp_from_query(query=new_count, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# new_pomdp.bel['cup'].p[10,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 'red'
# #new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,10] = 0.295 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,10] = 0.3 # Height in m

# new_pomdp.bel['cup'].p[15,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 'black'
# #new_pomdp.bel['cup'].feature_bels['volume']['bel'][15,10,10] = 0.310 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][15,10,10] = 0.31 # Height in m

# new_pomdp.bel['cup'].p[10,15,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 'green'
# #new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,15,10] = 0.400 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,15,10] = 0.32 # Height in m

# new_pomdp.bel['cup'].p[10,10,15] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 'yellow'
# #new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,15] = 0.250 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,15] = 0.33 # Height in m

# new_pomdp.bel['cup'].p[0,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 'red'
# #new_pomdp.bel['cup'].feature_bels['volume']['bel'][0,10,10] = 0.350 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][0,10,10] = 0.19 # Height in m

# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")


# # Avg Cup Volume single obj where red
# print("Avg Cup Vol Where red")
# new_where = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
# new_query = Query(obj_tp="cup", where_clause=new_where)

# new_map = Map(obj_tp='cup', map_feature='volume', query=new_query, map_tp='feature_scalar')

# new_agg = Aggregator(agg_tp='avg', symbolic_list=new_map)

# new_pomdp = gen_pomdp_from_query(query=new_agg, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)

# new_pomdp.bel['cup'].p[10,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,10] = 0.295 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,10] = 0.3 # Height in m

# new_pomdp.bel['cup'].p[15,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 'black'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][15,10,10] = 0.310 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][15,10,10] = 0.31 # Height in m

# new_pomdp.bel['cup'].p[10,15,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 'green'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,15,10] = 0.400 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,15,10] = 0.32 # Height in m

# new_pomdp.bel['cup'].p[10,10,15] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 'yellow'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,15] = 0.250 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][10,10,15] = 0.33 # Height in m

# new_pomdp.bel['cup'].p[0,10,10] = 0.95
# new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 'red'
# new_pomdp.bel['cup'].feature_bels['volume']['bel'][0,10,10] = 0.350 # Volume in liters 
# #new_pomdp.bel['cup'].feature_bels['height']['bel'][0,10,10] = 0.19 # Height in m

# symbolic_info = new_pomdp.make_symbolic()
# print("Symbolic Info: ", symbolic_info)
# print("\n\n")


# Avg Cup Volume where red + average bowl volume where blue
print("Avg Cup Vol Where red + avg bowl volume where blue")
new_where_cup = WhereClause(where_tp='feature_enum', obj_tp="cup", enum_feature='colour', enum_param='red')
new_query_cup = Query(obj_tp="cup", where_clause=new_where_cup)
new_map_cup = Map(obj_tp='cup', map_feature='volume', query=new_query_cup, map_tp='feature_scalar')
new_agg_cup = Aggregator(agg_tp='avg', symbolic_list=new_map_cup)

new_where_bowl = WhereClause(where_tp='feature_enum', obj_tp="bowl", enum_feature='colour', enum_param='blue')
new_query_bowl = Query(obj_tp="bowl", where_clause=new_where_bowl)
new_map_bowl = Map(obj_tp='bowl', map_feature='volume', query=new_query_bowl, map_tp='feature_scalar')
new_agg_bowl = Aggregator(agg_tp='avg', symbolic_list=new_map_bowl)

new_prim = Primitives(prim_tp='op', prim=new_agg_cup, prim2=new_agg_bowl, prim_op='plus')

new_pomdp = gen_pomdp_from_query(query=new_prim, pos=pos, ori=ori, trav_map_og_size=map_size, trav_map_og_res=og_resolution, configs=config)


# Cups
new_pomdp.bel['cup'].p[10,10,10] = 0.95
new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 'red'
new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,10] = 0.295 # Volume in liters 

new_pomdp.bel['cup'].p[15,10,10] = 0.95
new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 'black'
new_pomdp.bel['cup'].feature_bels['volume']['bel'][15,10,10] = 0.310 # Volume in liters 

new_pomdp.bel['cup'].p[10,15,10] = 0.95
new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 'green'
new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,15,10] = 0.400 # Volume in liters 

new_pomdp.bel['cup'].p[10,10,15] = 0.95
new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 'yellow'
new_pomdp.bel['cup'].feature_bels['volume']['bel'][10,10,15] = 0.250 # Volume in liters 

new_pomdp.bel['cup'].p[0,10,10] = 0.95
new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 'red'
new_pomdp.bel['cup'].feature_bels['volume']['bel'][0,10,10] = 0.350 # Volume in liters 


# Bowls
new_pomdp.bel['bowl'].p[20,20,10] = 0.95
new_pomdp.bel['bowl'].feature_bels['colour']['bel'][20,20,10] = 'blue'
new_pomdp.bel['bowl'].feature_bels['volume']['bel'][20,20,10] = 0.295 # Volume in liters 


new_pomdp.bel['bowl'].p[25,25,10] = 0.95
new_pomdp.bel['bowl'].feature_bels['colour']['bel'][25,25,10] = 'green'
new_pomdp.bel['bowl'].feature_bels['volume']['bel'][25,25,10] = 1.2 # Volume in liters 



symbolic_info = new_pomdp.make_symbolic()
print("Symbolic Info: ", symbolic_info)
print("\n\n")
