import numpy as np

from RoboInfoGather.pomdp import *
from RoboInfoGather.MCTS_planner import Loc
from RoboInfoGather.observation_utils import *
from RoboInfoGather.map_utils import *
from RoboInfoGather.dsl import *

from RoboInfoGather.synthesis.synthesis import *

import copy

from matplotlib import pyplot as plt

def gen_prog_from_nl(nl):
    # Call Synth
    synthesizer = Synthesizer(
        config_file_path="RoboInfoGather/synthesis/synthesis_config.yaml",
        example_file_path="RoboInfoGather/synthesis/examples.yaml",
        preamble_file_path="RoboInfoGather/synthesis/prompt_preamble.txt",
    )

    result = synthesizer.generate(nl)
    new_prog = result.get_program_object()

    return new_prog

def get_objects_and_features_helper(component):
    # Start with list and then unify
    if type(component) is Query:
        obj_feat = [(component.obj_tp, None, None)]
        obj_feat += get_objects_and_features_helper(component.where_clause)
        return obj_feat
    elif type(component) is WhereClause:
        if component.where_tp == "feature_enum":
            return [(component.obj_tp, component.enum_feature, "feature_enum")]

        elif component.where_tp == "feature_scalar":
            return [(component.obj_tp, component.scalar_feature, "feature_scalar")]

        elif component.where_tp == "max":
            return [(component.obj_tp, component.scalar_feature, "feature_scalar")]

        elif component.where_tp == "min":
            return [(component.obj_tp, component.scalar_feature, "feature_scalar")]

        elif component.where_tp == "spatial_rel":
            return [(component.obj_tp2, None, None)]

        elif component.where_tp == "and":
            obj_feat = get_objects_and_features_helper(component.sub_where_clause[0])
            obj_feat += get_objects_and_features_helper(component.sub_where_clause[1])
            return obj_feat

        elif component.where_tp == "or":
            obj_feat = get_objects_and_features_helper(component.sub_where_clause[0])
            obj_feat += get_objects_and_features_helper(component.sub_where_clause[1])
            return obj_feat

        elif component.where_tp == "not":
            obj_feat = get_objects_and_features_helper(component.sub_where_clause[0])
            return obj_feat

        elif component.where_tp == "true":
            return []

    elif type(component) is Map or type(component) is Count:
        return get_objects_and_features_helper(component.query)

    elif type(component) is Primitives:
        if component.prim_tp == 'real':
            return []
        elif component.prim_tp == "op":
            obj_feat = get_objects_and_features_helper(component.prim_tp)
            obj_feat += get_objects_and_features_helper(component.prim_tp2)
            return obj_feat

        else:
            return get_objects_and_features_helper(component.prim_tp)

    elif type(component) is GetNth or type(component) is Aggregator:
        return get_objects_and_features_helper(component.list)


def get_objects_and_features(query):
    # Start with list and then unify
    obj_feat_list = get_objects_and_features_helper(query)

    # Unify
    obj_feat_dict = {}
    print(obj_feat_list)
    for obj, feat, tp in obj_feat_list:
        if obj in obj_feat_dict:
            if feat not in obj_feat_dict[obj]:
                enc_feat = {'name': feat, 'tp': tp}
                obj_feat_dict[obj].append(enc_feat)
        else:
            enc_feat = {'name': feat, 'tp': tp}
            obj_feat_dict[obj] = [enc_feat]

    return obj_feat_dict

def gen_pomdp_from_query(query, pos, yaw, trav_map_og_size, trav_map_og_res, configs, prev_pomdp=None, gen_inform_priors=None):
    # Recurse on cases where "query" is aggregator, map, primitives, getnth, or count
    if type(query) is Map:
        # Add map feature to where clause for feature extraction
        query_to_send = copy.deepcopy(query.query)
        if query.map_tp == 'feature_scalar':
            temp_where = WhereClause(where_tp=query.map_tp, obj_tp=query.obj_tp, scalar_feature=query.map_feature)
        else:
            temp_where = WhereClause(where_tp=query.map_tp, obj_tp=query.obj_tp, enum_feature=query.map_feature)
        temp_where_and = WhereClause(where_tp='and', obj_tp=query.obj_tp, sub_where_clause=[temp_where, query_to_send.where_clause])
        query_to_send.where_clause = temp_where_and
        return gen_pomdp_from_query(query_to_send, pos, yaw, trav_map_og_size, trav_map_og_res, configs, prev_pomdp, gen_inform_priors)
    if type(query) is Primitives:
        # Need to combine pomdps for both prim sides
        new_pomdp = gen_pomdp_from_query(query.prim, pos, yaw, trav_map_og_size, trav_map_og_res, configs, prev_pomdp, gen_inform_priors)
        return gen_pomdp_from_query(query.prim2, pos, yaw, trav_map_og_size, trav_map_og_res, configs, new_pomdp, gen_inform_priors)
    if type(query) is GetNth:
        return gen_pomdp_from_query(query.list, pos, yaw, trav_map_og_size, trav_map_og_res, configs, prev_pomdp, gen_inform_priors)
    if type(query) is Count:
        return gen_pomdp_from_query(query.query, pos, yaw, trav_map_og_size, trav_map_og_res, configs, prev_pomdp, gen_inform_priors)
    if type(query) is Aggregator:
        return gen_pomdp_from_query(query.list, pos, yaw, trav_map_og_size, trav_map_og_res, configs, prev_pomdp, gen_inform_priors)

    theta = yaw 
    robot_init_loc = Loc(pos[0], pos[1], theta)

    if prev_pomdp == None:
        # Search Query and get object type list
        # (obj_tp, num, thresh, [relevant_features], priors)
        threshold = query.threshold
        num = query.limit

        obj_feat_dict = get_objects_and_features(query)

        # Make list
        obj_tp_list = []
        for obj in obj_feat_dict:
            obj_tp_list.append((obj, num, threshold, obj_feat_dict[obj], None))

        if gen_inform_priors != None:
            assert False # PRIORS?
        else:
            new_pomdp = POMDP(query, robot_init_loc, obj_tp_list, trav_map_og_size, trav_map_og_res, configs)

        return new_pomdp

    else: # Build off of old pomdp
        threshold = query.threshold
        num = query.limit

        obj_feat_dict = get_objects_and_features(query)

        # Make list and unify with old
        obj_tp_list = []
        for obj in obj_feat_dict:
            if obj not in prev_pomdp.bel:
                # Get map params based off of current params and obj_tp
                map_params = get_map_params(obj, prev_pomdp.trav_map_original_size, prev_pomdp.trav_map_original_resolution)

                if gen_inform_priors == None:
                    prev_pomdp.bel[obj] = ObjTpBel(num, threshold, map_params, configs, obj_feat_dict[obj])
                else:
                    prev_pomdp.bel[obj] = gen_inform_priors

                prev_pomdp.reward_funcs[obj] = RewardFunc(map_params, prev_pomdp.camera_params, prev_pomdp.rf_params)

                if prev_pomdp.configs['bel_params']['visualize']:
                    # Add figures
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    plt.ion()
                    plt.show()
                    prev_pomdp.figures[obj] = (fig, ax)

            else:
                # Check for features
                for feature in obj_feat_dict[obj]:
                    if feature not in prev_pomdp.bel[obj].feature_bels:
                        # Make a new feature belief
                        feature_dict = {}
                        p = np.copy(prev_pomdp.bel[obj].p)
                        if feature['tp'] == "feature_scalar":
                            feature_dict = {'bel': np.copy(p), "tp" : feature['tp'], "vals": np.zeros_like(p)}
                        elif feature['tp'] == "feature_enum":
                            # For features, we want to keep around names like colour = red
                            x_dim, y_dim, z_dim = p.shape
                            val_z = np.array(['' for _ in range(z_dim)], dtype=object)
                            val_y = np.array([val_z for _ in range(y_dim)], dtype=object)
                            val = np.array([val_y for _ in range(x_dim)], dtype=object)

                            # Should have same shape
                            assert val.shape == p.shape

                            feature_dict = {'bel': np.copy(p), "tp" : feature['tp'], "vals": val}

                        prev_pomdp.bel[obj].feature_bels[feature['name']] = feature_dict

        return prev_pomdp
