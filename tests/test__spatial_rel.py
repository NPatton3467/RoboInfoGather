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

def check_eq_tup(tup1, tup2):
    if len(tup1) != len(tup2):
        print("Different Length")
        return False

    for x in range(len(tup1)):
        if round(float(tup1[x]),1) != round(float(tup2[x]), 1):
            print("Not Equal: ", float(tup1[x]), float(tup2[x]))
            return False

    return True

def check_eq(dict1, dict2):
    equal = True
    for key in dict1:
        if key not in dict2:
            print("Broke Key not in Dict 2")
            print(dict1, dict2, key)
            return False
        elif type(dict1[key]) != type(dict2[key]):
            print("Broke dict1[key] not eq dict2[key]")
            print(dict1[key], dict2[key])
            return False
        elif type(dict1[key]) is dict:
            equal = check_eq(dict1[key], dict2[key])
            if not equal:
                print("Broke dict1[key] not eq dict2[key]")
                print(dict1[key], dict2[key])
                return False
        elif type(dict1[key]) is tuple:
            equal = check_eq_tup(dict1[key], dict2[key])
            if not equal:
                print("Broke dict1[key] not eq dict2[key]")
                print(dict1[key], dict2[key])
                return False
        else: 
            equal = (dict1[key] == dict2[key])
            if not equal:
                print("Broke dict1[key] not eq dict2[key]")
                print(dict1[key], dict2[key], key)

                return False

    return equal

# Load the config
config_filename = os.path.join(f"/robodata/user_data/npatt/explore-eqa/RoboInfoGather/info_gather_test.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

# Get Position/orientation
pos = np.array([0,0,0])
angle = 0
dim = np.array([100, 100, 50])
resolution = 0.1
vol_origin = np.array([0,0,0])

# # And where clause -- red cups on tables
print("Where colour = red and on table")
for i in range(10): # Test 10 times
    print("Check: ", i)
    sub_where_colour = WhereClause(where_tp='feature_enum', obj_tp='cup', enum_feature='colour', enum_param='red')

    sub_where_on = WhereClause(where_tp='spatial_rel', obj_tp='cup', spatial_relation='on', obj_tp2='table')

    new_where = WhereClause(where_tp='and', obj_tp='cup', sub_where_clause=[sub_where_colour, sub_where_on])

    new_query = Query(obj_tp="cup", where_clause=new_where)
    #print(new_query.pretty_str())

    new_pomdp = gen_pomdp_from_query(
            query=new_query,
            pos=pos,
            yaw=angle,
            trav_map_og_dim=dim,
            trav_map_og_res=resolution,
            vol_origin=vol_origin,
            configs=config
        )

    new_pomdp.bel['cup'].p[10,10,10] = 0.95
    new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,10] = 0.95
    new_pomdp.bel['cup'].feature_bels['colour']['vals'][10,10,10] = 'red'

    new_pomdp.bel['cup'].p[15,10,10] = 0.95
    new_pomdp.bel['cup'].feature_bels['colour']['bel'][15,10,10] = 0.95
    new_pomdp.bel['cup'].feature_bels['colour']['vals'][15,10,10] = 'black'

    new_pomdp.bel['cup'].p[10,15,10] = 0.95
    new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,15,10] = 0.95
    new_pomdp.bel['cup'].feature_bels['colour']['vals'][10,15,10] = 'green'

    new_pomdp.bel['cup'].p[10,10,15] = 0.95
    new_pomdp.bel['cup'].feature_bels['colour']['bel'][10,10,15] = 0.95
    new_pomdp.bel['cup'].feature_bels['colour']['vals'][10,10,15] = 'yellow'

    new_pomdp.bel['cup'].p[0,10,10] = 0.95
    new_pomdp.bel['cup'].feature_bels['colour']['bel'][0,10,10] = 0.95
    new_pomdp.bel['cup'].feature_bels['colour']['vals'][0,10,10] = 'red'

    new_pomdp.bel['table'].p[0,1,1] = 0.95

    symbolic_info = new_pomdp.make_symbolic()
    #print("Symbolic Info: ", symbolic_info)
    result = new_query.execute(symbolic_info)
    #print("\nResult:\n", result)
    #print("\n\n")

    true_res = {'cup': {0: {'colour': 'red', 'location': (0.0, 0.8, 1.1), 'on': 'table_0'}}}
    if not check_eq(result, true_res):
        print(result)
        print(type(result))
        print(true_res)
        print(type(true_res))
        assert False


# # And where clause -- lamps on beds
print("Where near(lamp, bed)")
for i in range(10): # Test 10 times
    print("Check: ", i)
    where_on = WhereClause(where_tp='spatial_rel', obj_tp='lamp', spatial_relation='near', obj_tp2='bed')

    new_query = Query(obj_tp="lamp", where_clause=where_on)
    #print(new_query.pretty_str())

    new_pomdp = gen_pomdp_from_query(
            query=new_query,
            pos=pos,
            yaw=angle,
            trav_map_og_dim=dim,
            trav_map_og_res=resolution,
            vol_origin=vol_origin,
            configs=config
        )

    #print(new_pomdp.bel['lamp'].p.shape)
    #print(new_pomdp.bel['bed'].p.shape)
    new_pomdp.bel['lamp'].p[1,1,1] = 0.95
    new_pomdp.bel['lamp'].p[10,10,1] = 0.95
    new_pomdp.bel['lamp'].p[20,20,1] = 0.95
    new_pomdp.bel['lamp'].p[30,30,1] = 0.95
    new_pomdp.bel['lamp'].p[40,40,1] = 0.95

    new_pomdp.bel['bed'].p[1,1,1] = 0.95

    symbolic_info = new_pomdp.make_symbolic()
    #print("Symbolic Info: ", symbolic_info)
    result = new_query.execute(symbolic_info)
    #print("\nResult:\n", result)
    #print("\n\n")

    true_res = {'lamp': {0: {'location': (0.2, 0.2, 0.5), 'near': 'bed_0'}, 1: {'location': (2.0, 2.0, 0.5), 'near': 'bed_0'}, 2: {'location': (4.0, 4.0, 0.5), 'near': 'bed_0'}}, 'bed': {0: {'location': (1.9, 1.9, 0.6)}}}
    if not check_eq(result, true_res):
        print(result)
        print(type(result))
        print(true_res)
        print(type(true_res))
        assert False
