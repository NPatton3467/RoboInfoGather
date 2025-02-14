import copy
import numpy as np
import openai
from openai import OpenAI

f = open('/robodata/user_data/npatt/explore-eqa/RoboInfoGather/openaikey.txt', 'r')
openai_api_key = f.read().rstrip('\n')
f.close()

def eval_feature_equality(found_feature_val, comp, real_feature_val):
    f = open('./RoboInfoGather/eval_feature_pre_prompt.txt', 'r')
    pre_prompt = f.read()
    f.close()

    post_prompt = f"\nNow given feature (1) {found_feature_val}, and feature (2) {real_feature_val}, evaluate whether feature (1) is equivalent to feature (2). Please answer with only equal or not equal.\nAnswer:\n"

    prompt = pre_prompt + post_prompt

    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"{prompt}"}],
        stream=False,
        temperature=0.0
    )

    response = response.choices[0].message.content

    truth_val = False
    if (response.lower() == "equal" and comp == "==") or\
            (response.lower() == "not equal" and comp == "!="):
        truth_val = True

    return truth_val


class Prog:
    def __init__(self, expressions):
        self.expressions = expressions

    def pretty_str(self):
        res = ''
        for exp in self.expressions:
            res += f"{exp.pretty_str()}\n"

        return res

    def execute(self, symbolic_info):
        results = []

        for exp in self.expressions:
            results.append(exp.execute(symbolic_info))

        return results 


class Map:
    def __init__(self, obj_tp, map_feature, query, map_tp):
        self.map_tp = map_tp
        self.obj_tp = obj_tp
        self.map_feature = map_feature
        self.query = query

    def pretty_str(self):
        return f"map({self.obj_tp}, {self.map_feature}, {self.query.pretty_str()})"

    def execute(self, symbolic_info):
        # Query must have been executed in real world to get symbolic results
        query_result = self.query.execute(symbolic_info)

        result = {}
        if self.obj_tp in query_result:
            local_dict = {}
            for obj_inst in query_result[self.obj_tp]:
                local_dict[obj_inst] = {}
                local_dict[obj_inst][self.map_feature] = query_result[self.obj_tp][obj_inst][self.map_feature]

            result[self.obj_tp] = local_dict

        return result

class Primitives:
    def __init__(self, prim_tp, prim, prim2=None, prim_op=None):
        self.prim_tp = prim_tp
        self.prim = prim
        self.prim2 = prim2
        self.prim_op = prim_op

    def pretty_str(self):
        if self.prim_tp == 'real':
            return f"{self.prim}"
        elif self.prim_tp == "op":
            op = ""
            if self.prim_op == "plus":
                op = "+"
            elif self.prim_op == "minus":
                op = "-"
            elif self.prim_op == "mul":
                op = "*"
            elif self.prim_op == "div":
                op = "/"

            return f"{self.prim.pretty_str()} {op} {self.prim2.pretty_str()}"

        else:
            return f"{self.prim.pretty_str()}"

    def execute(self, symbolic_info):
        if self.prim_tp == "real":
            result = self.prim
        elif self.prim_tp == "op":
            # Execute left and right primitives
            left = 0
            right = 0
            if type(self.prim) in [GetNth, Count, Aggregator]:
                prim_result = self.prim.execute(symbolic_info)

                left = prim_result
            else:
                left = self.prim
            
            if type(self.prim2) in [GetNth, Count, Aggregator]:
                prim2_result = self.prim2.execute(symbolic_info)

                right = prim2_result
            else:
                right = self.prim2

        
            # Perform the operation
            if self.prim_op == "plus":
                result = left + right
            elif self.prim_op == "minus":
                result = left - right
            elif self.prim_op == "mul":
                result = left * right
            elif self.prim_op == "div":
                result = left / right

        else:
            if type(self.prim) in [GetNth, Count, Aggregator]:
                prim_result = self.prim.execute()

                result = prim_result

            else:
                result = self.prim

        return result


class GetNth:
    def __init__(self, symbolic_list, index):
        self.list = symbolic_list
        self.index = index 

    def pretty_str(self):
        return f"getNth({self.list.pretty_str()}, {self.index})"

    def execute(self, symbolic_info):
        result = {}
        # List must have been evaluated
        print("In GetNth... Symbolic Info: ", symbolic_info)
        list_result = self.list.execute(symbolic_info)

        print("In GetNth... self.list.result: ", list_result)
        obj_tp = list(list_result.keys())[0]
        if obj_tp in list_result:
            print("In GetNth... obj_tp: ", obj_tp)
            key = list(list_result[obj_tp].keys())[self.index]
            print("In GetNth... key: ", key)
            if key in list_result[obj_tp]:
                result[obj_tp] = list_result[obj_tp][key]

        return result


class Count:
    def __init__(self, query, obj_tp):
        self.query = query
        self.obj_tp = obj_tp

    def pretty_str(self):
        return f"count({self.query.pretty_str()}, {self.obj_tp})"

    def execute(self, symbolic_info):
        # Query must have been executed in the real world
        result = {}
        query_result = self.query.execute(symbolic_info)

        if self.obj_tp in query_result:
            result[self.obj_tp] = {"Count" : len(query_result[self.obj_tp])}

        return result


class Aggregator:
    def __init__(self, agg_tp, symbolic_list):
        self.agg_tp = agg_tp
        self.list = symbolic_list

    def pretty_str(self):
        return f"{self.agg_tp}({self.list.pretty_str()})"

    def execute(self, symbolic_info):
        # List must have been completed
        list_result = self.list.execute(symbolic_info)

        result = None
        if self.agg_tp == "sum":
            tsum = 0
            obj_tp = list(list_result.keys())[0]

            if obj_tp in list_result:
                for inst in list_result[obj_tp]:
                    feature = list(list_result[obj_tp][inst].keys())[0]
                    tsum += list_result[obj_tp][inst][feature]

            result = {obj_tp : {f'Total {feature}' : tsum}}

        elif self.agg_tp == "avg":
            avg = 0
            obj_tp = list(list_result.keys())[0]

            if obj_tp in list_result:
                for inst in list_result[obj_tp]:
                    feature = list(list_result[obj_tp][inst].keys())[0]
                    avg += list_result[obj_tp][inst][feature]

            result = {obj_tp : {f'Average {feature}' : avg/len(list_result[obj_tp])}}

        elif self.agg_tp == "min":
            tmin = -1
            obj_tp = list(list_result.keys())[0]

            if obj_tp in list_result:
                for inst in list_result[obj_tp]:
                    feature = list(list_result[obj_tp][inst].keys())[0]
                    if list_result[obj_tp][inst][feature] < tmin or tmin == -1:
                        tmin += list_result[obj_tp][inst][feature]

            result = {obj_tp : {f'Minimum {feature}' : tmin}}

        elif self.agg_tp == "max":
            tmax = -1
            obj_tp = list(list_result.keys())[0]

            if obj_tp in list_result:
                for inst in list_result[obj_tp]:
                    feature = list(list_result[obj_tp][inst].keys())[0]
                    if list_result[obj_tp][inst][feature] > tmax or tmax == -1:
                        tmax += list_result[obj_tp][inst][feature]

            result = {obj_tp : {f'Maximum {feature}' : tmax}}

        return result


class Query:
    def __init__(self, obj_tp, where_clause, limit=-1, threshold=0.9):
        self.obj_tp = obj_tp
        self.where_clause = where_clause
        self.limit = limit
        self.threshold = threshold

    def pretty_str(self):
        res = f'find ({self.obj_tp}) where ({self.where_clause.pretty_str()})'

        if self.limit > 0:
            assert False # This won't work, need query return to be a dict
            res += f" [limit {self.limit}]"

        return res

    def execute(self, symbolic_info):
        result = self.where_clause.filter(copy.deepcopy(symbolic_info))
        
        print("Current result in Query:\n", result)
        if self.obj_tp in result and len(result[self.obj_tp]) > self.limit and self.limit > 0:
            result[self.obj_tp] = result[self.obj_tp][0:self.limit]

        return result


class WhereClause:
    def __init__(self, where_tp, obj_tp, sub_where_clause=None, obj_tp2=None, 
        scalar_feature=None, scalar_param=None, scalar_comparator=None, enum_feature=None, enum_param=None,
        spatial_relation=None, is_temp=False):

        self.where_tp = where_tp
        self.obj_tp = obj_tp
        self.sub_where_clause = sub_where_clause
        self.obj_tp2 = obj_tp2
        self.scalar_feature = scalar_feature
        self.scalar_param = scalar_param
        self.scalar_comparator = scalar_comparator
        self.enum_feature = enum_feature
        self.enum_param = enum_param
        self.spatial_relation = spatial_relation

        # Assertions for well-formedness
        if not is_temp:
            if self.where_tp == "and" or self.where_tp == "or":
                assert len(self.sub_where_clause) == 2
                assert type(self.sub_where_clause[0]) is WhereClause
                assert type(self.sub_where_clause[1]) is WhereClause

            if self.where_tp == "not":
                assert len(self.sub_where_clause) == 1
                assert type(self.sub_where_clause[0]) is WhereClause

            if self.where_tp == "feature_enum":
                assert self.enum_feature is not None and self.enum_param is not None

            if self.where_tp == "feature_scalar":
                assert self.scalar_comparator is not None and self.scalar_feature is not None and self.scalar_param is not None

            if self.where_tp == "max" or self.where_tp == "min":
                assert self.scalar_feature is not None

            if self.where_tp == "spatial_rel":
                assert self.spatial_relation is not None and self.obj_tp2 is not None



    def pretty_str(self):
        if self.where_tp == "feature_enum":
            return f"{self.enum_feature}({self.obj_tp}) = {self.enum_param}"

        elif self.where_tp == "feature_scalar":
            scalar_comp = ""
            if self.scalar_comparator == "Lt":
                scalar_comp = "<"
            elif self.scalar_comparator == "Leq":
                scalar_comp = "<="
            elif self.scalar_comparator == "Eq":
                scalar_comp = "="
            elif self.scalar_comparator == "Geq":
                scalar_comp = ">="
            elif self.scalar_comparator == "Gt":
                scalar_comp = ">"

            return f"{self.scalar_feature}({self.obj_tp}) {scalar_comp} {self.scalar_param}"

        elif self.where_tp == "max":
            return f"max({self.scalar_feature}({self.obj_tp}))"

        elif self.where_tp == "min":
            return f"min({self.scalar_feature}({self.obj_tp}))"

        elif self.where_tp == "spatial_rel":
            return f"{self.spatial_relation}({self.obj_tp}, {self.obj_tp2})"

        elif self.where_tp == "and":
            return f"{self.sub_where_clause[0].pretty_str()} /\ {self.sub_where_clause[1].pretty_str()}"

        elif self.where_tp == "or":
            return f"{self.sub_where_clause[0].pretty_str()} \/ {self.sub_where_clause[1].pretty_str()}"

        elif self.where_tp == "not":
            return f"!({self.sub_where_clause[0].pretty_str()})"

        elif self.where_tp == "true":
            return "true"

    def filter(self, symbolic_info):
        ret_symb_info = {}
        if self.where_tp == 'true':
            ret_symb_info = copy.deepcopy(symbolic_info)
        elif self.where_tp == "feature_enum" or self.where_tp =="feature_scalar":
            comp = "=="
            if self.scalar_comparator == "Lt":
                comp = "<"
            elif self.scalar_comparator == "Leq":
                comp = "<="
            elif self.scalar_comparator == "Geq":
                comp = ">="
            elif self.scalar_comparator == "Gt":
                comp = ">"
            elif self.scalar_comparator == "Neq":
                comp = "!="

            temp_dict = {}
            if self.obj_tp in symbolic_info:
                for inst in symbolic_info[self.obj_tp]:
                    if self.where_tp == "feature_enum" and (comp == "==" or comp == "!="):
                        if self.enum_feature in symbolic_info[self.obj_tp][inst]:
                            # Use LLM to evaluate feature
                            eval_true = eval_feature_equality(symbolic_info[self.obj_tp][inst][self.enum_feature], comp, self.enum_param)
                            if eval_true:
                                # Make proper feature based on enum_param
                                temp_inst_dict = symbolic_info[self.obj_tp][inst]
                                if comp == "==":
                                    temp_inst_dict[self.enum_feature] = self.enum_param

                                temp_dict[inst] = temp_inst_dict

                    elif self.where_tp == "feature_enum":
                        if self.enum_feature in symbolic_info[self.obj_tp][inst] and\
                            eval(f"'{symbolic_info[self.obj_tp][inst][self.enum_feature]}' {comp} '{self.enum_param}'"):
                            
                                temp_dict[inst] = symbolic_info[self.obj_tp][inst]

                    elif self.where_tp == "feature_scalar":
                        if self.scalar_feature in symbolic_info[self.obj_tp][inst] and\
                          eval(f"{symbolic_info[self.obj_tp][inst][self.scalar_feature]} {comp} {self.scalar_param}"):
                            
                            temp_dict[inst] = symbolic_info[self.obj_tp][inst]

            ret_symb_info[self.obj_tp] = temp_dict                

        elif self.where_tp == "max":
            temp_obj = None
            max_val = -1
            if self.obj_tp in symbolic_info:
                for inst in symbolic_info[self.obj_tp]:
                    if self.scalar_feature in symbolic_info[self.obj_tp][inst] and\
                     (symbolic_info[self.obj_tp][inst][self.scalar_feature] > max_val or max_val == -1):
                        temp_obj = {inst: symbolic_info[self.obj_tp][inst]}
                        max_val = symbolic_info[self.obj_tp][inst][self.scalar_feature]

            ret_symb_info[self.obj_tp] = temp_obj

        elif self.where_tp == "min":
            temp_obj = None
            min_val = -1
            if self.obj_tp in symbolic_info:
                for inst in symbolic_info[self.obj_tp]:
                    if self.scalar_feature in symbolic_info[self.obj_tp][inst] and\
                     (symbolic_info[self.obj_tp][inst][self.scalar_feature] < min_val or min_val == -1):
                        temp_obj = {inst: symbolic_info[self.obj_tp][inst]}
                        min_val = symbolic_info[self.obj_tp][inst][self.scalar_feature]

            ret_symb_info[self.obj_tp] = temp_obj

        elif self.where_tp == "spatial_rel":
            temp_dict1 = {}
            temp_dict2 = {}
            if self.obj_tp in symbolic_info:
                print("Current Symbolic Info (spatial_rel):\n", symbolic_info)
                for inst1 in symbolic_info[self.obj_tp]:
                    if self.obj_tp2 in symbolic_info:
                        for inst2 in symbolic_info[self.obj_tp2]:
                            # Query LLM for spatial rel
                            f = open('./RoboInfoGather/spatial_rel_pre_prompt.txt', 'r')
                            pre_prompt = f.read()
                            f.close()
                            
                            loc1 = symbolic_info[self.obj_tp][inst1]['location']
                            loc2 = symbolic_info[self.obj_tp2][inst2]['location']
                            prompt = pre_prompt + f"\n\nNow given object (1) of type {self.obj_tp} with location {loc1}, and object (2) of type {self.obj_tp2} with location {loc2}. Is object (1) {self.spatial_relation} object (2)? Please answer with only True or False.\nAnswer:" 
                            #print("\nSpatial Relation Locations:\n", loc1, "\n", loc2)
                            client = OpenAI(api_key=openai_api_key)
                            response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[{"role": "user", "content": f"{prompt}"}],
                                stream=False,
                                temperature=0.0
                            )

                            # Extract response
                            response = response.choices[0].message.content
                            response = response.strip(' \n')
                            #print("\nSpatial Relation Response:\n", response)
                            if response == "True" or response == "true":
                                # Append "spatial_rel" to the temp_dist 1
                                temp_inst_dict1 = symbolic_info[self.obj_tp][inst1]
                                temp_inst_dict1[self.spatial_relation] = str(self.obj_tp2) + "_" + str(inst2)
                                temp_dict1[inst1] = temp_inst_dict1
                                temp_dict2[inst2] = symbolic_info[self.obj_tp2][inst2]

            ret_symb_info[self.obj_tp] = temp_dict1
            ret_symb_info[self.obj_tp2] = temp_dict2

        elif self.where_tp == "and":
            # Check if sub_where is binary predicate -- need to evaluate first
            if self.sub_where_clause[1].where_tp == "spatial_rel":
                ret_symb_info = self.sub_where_clause[1].filter(symbolic_info)
                ret_symb_info = self.sub_where_clause[0].filter(ret_symb_info)
            else:
                ret_symb_info = self.sub_where_clause[0].filter(symbolic_info)
                ret_symb_info = self.sub_where_clause[1].filter(ret_symb_info)

        elif self.where_tp == "or":
            left_symb_info = self.sub_where_clause[0].filter(symbolic_info)
            right_symb_info = self.sub_where_clause[1].filter(symbolic_info)

            
            # Combine
            ret_symb_info = {}
            for obj_tp in symbolic_info:
                if obj_tp in left_symb_info or obj_tp in right_symb_info:
                    temp_dict = {}
                    if obj_tp in symbolic_info:
                        for inst in symbolic_info[obj_tp]:
                            inleft = False
                            inright = False

                            if obj_tp in left_symb_info and inst in left_symb_info[obj_tp]:
                                inleft = True

                            if obj_tp in right_symb_info and inst in right_symb_info[obj_tp]:
                                inright = True

                            if inleft or inright:
                                temp_dict[inst] = symbolic_info[obj_tp][inst]
                    
                    ret_symb_info[obj_tp] = temp_dict

        elif self.where_tp == "not":
            true_ret_info = self.sub_where_clause[0].filter(symbolic_info)

            # Compare to ret symb info
            # Remove version that are in true_ret_info
            keep_obj = {}
            if self.obj_tp in symbolic_info:
                for obj_dict in symbolic_info[self.obj_tp]:
                    in_true = False

                    if self.obj_tp in true_ret_info:
                        for t_obj_dict in true_ret_info[self.obj_tp]:
                            if obj_dict == t_obj_dict:
                                in_true = True

                    if not in_true:
                        keep_obj[obj_dict] = symbolic_info[self.obj_tp][obj_dict]

                ret_symb_info[self.obj_tp] = keep_obj

        return ret_symb_info
