import copy
import numpy as np

import openai
from openai import OpenAI
f = open('./RoboInfoGather/openaikey.txt', 'r')
openai_api_key = f.read().rstrip('\n')
f.close()

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

        self.result = {}

    def pretty_str(self):
        return f"map({self.obj_tp}, {self.map_feature}, {self.query.pretty_str()})"

    def execute(self, symbolic_info):
        # Query must have been executed in real world to get symbolic results
        if self.query.result == None:
            self.query.execute(symbolic_info)

        if self.result == {}:
            if self.obj_tp in self.query.result:
                for obj_inst in self.query.result[self.obj_tp]:
                    self.result[obj_inst] = self.query.result[self.obj_tp][obj_inst][self.map_feature]

        return self.result

class Primitives:
    def __init__(self, prim_tp, prim, prim2=None, prim_op=None):
        self.prim_tp = prim_tp
        self.prim = prim
        self.prim2 = prim2
        self.prim_op = prim_op

        # Result to be filled
        self.result = None

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
        if self.result == None:
            if self.prim_tp == "real":
                self.result = self.prim
            elif self.prim_tp == "op":
                # Execute left and right primitives
                left = 0
                right = 0
                if type(self.prim) in [GetNth, Count, Aggregator]:
                    if self.prim.result == None:
                        self.prim.execute(symbolic_info)

                    left = self.prim.result
                else:
                    left = self.prim
                
                if type(self.prim2) in [GetNth, Count, Aggregator]:
                    if self.prim2.result == None:
                        self.prim2.execute(symbolic_info)

                    right = self.prim2.result
                else:
                    right = self.prim2

            
                # Perform the operation
                if self.prim_op == "plus":
                    self.result = left + right
                elif self.prim_op == "minus":
                    self.result = left - right
                elif self.prim_op == "mul":
                    self.result = left * right
                elif self.prim_op == "div":
                    self.result = left / right

            else:
                if type(self.prim) in [GetNth, Count, Aggregator]:
                    if self.prim.result == None:
                        self.prim.execute()

                    self.result = self.prim.result

                else:
                    self.result = self.prim

        return self.result


class GetNth:
    def __init__(self, symbolic_list, index):
        self.list = symbolic_list
        self.index = index 

        # Result
        self.result = None

    def pretty_str(self):
        return f"getNth({self.list.pretty_str()}, {self.index})"

    def execute(self, symbolic_info):
        # List must have been evaluated
        if self.list.result == {}:
            self.list.execute(symbolic_info)

        if self.result == None:
            key = list(self.list.result.keys())[self.index]
            if key in self.list.result:
                self.result = self.list.result[key]

        return self.result


class Count:
    def __init__(self, query, obj_tp):
        self.query = query
        self.obj_tp = obj_tp

        # Result
        self.result = None

    def pretty_str(self):
        return f"count({self.query.pretty_str()}, {self.obj_tp})"

    def execute(self, symbolic_info):
        # Query must have been executed in the real world
        if self.query.result == None:
            self.query.execute(symbolic_info)

        if self.result == None:
            if self.obj_tp in self.query.result:
                self.result = len(self.query.result[self.obj_tp])

        return self.result


class Aggregator:
    def __init__(self, agg_tp, symbolic_list):
        self.agg_tp = agg_tp
        self.list = symbolic_list

        # Result
        self.result = None

    def pretty_str(self):
        return f"{self.agg_tp}({self.list.pretty_str()})"

    def execute(self, symbolic_info):
        # List must have been completed
        if self.list.result == {}:
            self.list.execute(symbolic_info)

        if self.result == None:
            if self.agg_tp == "sum":
                tsum = 0
                
                for key in self.list.result:
                    tsum += self.list.result[key]

                self.result = tsum

            elif self.agg_tp == "avg":
                avg = 0

                for key in self.list.result:
                    avg += self.list.result[key]

                self.result = avg/len(self.list.result)

            elif self.agg_tp == "min":
                tmin = -1

                for key in self.list.result:
                    if self.list.result[key] < tmin or tmin == -1:
                        tmin = self.list.result[key]

                self.result = tmin

            elif self.agg_tp == "max":
                tmax = -1

                for key in self.list.result:
                    if self.list.result[key] > tmax or tmax == -1:
                        tmax = self.list.result[key]

                self.result = tmax

        return self.result


class Query:
    def __init__(self, obj_tp, where_clause, limit=-1, threshold=0.9):
        self.obj_tp = obj_tp
        self.where_clause = where_clause
        self.limit = limit
        self.threshold = threshold

        # Result of executing Query
        self.result = None

    def pretty_str(self):
        res = f'find ({self.obj_tp}) where ({self.where_clause.pretty_str()})'

        if self.limit > 0:
            assert False # This won't work, need query return to be a dict
            res += f" [limit {self.limit}]"

        return res

    def execute(self, symbolic_info):
        if self.result == None:
            self.result = self.where_clause.filter(copy.deepcopy(symbolic_info))

            if self.obj_tp in self.result and len(self.result[self.obj_tp]) > self.limit and self.limit > 0:
                self.result[self.obj_tp] = self.result[self.obj_tp][0:self.limit]

        return self.result


class WhereClause:
    def __init__(self, where_tp, obj_tp, sub_where_clause=None, obj_tp2=None, 
        scalar_feature=None, scalar_param=None, scalar_comparator=None, enum_feature=None, enum_param=None,
        spatial_relation=None):

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
                    if self.where_tp == "feature_enum":
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
                for inst1 in symbolic_info[self.obj_tp]:
                    if self.obj_tp2 in symbolic_info:
                        for inst2 in symbolic_info[self.obj_tp2]:
                            # Query LLM for spatial rel
                            loc1 = symbolic_info[self.obj_tp][inst1]['location']
                            loc2 = symbolic_info[self.obj_tp2][inst2]['location']
                            prompt = f"Give object (1) of type {self.obj_tp} with location {loc1}, and object (2) of type {self.obj_tp2} with location {loc2}. Is object (1) {self.spatial_relation} object (2)? Please answer with only True or False." 
                            client = OpenAI(api_key=openai_api_key)
                            response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[{"role": "user", "content": f"{prompt}"}],
                                stream=False,
                                temperature=0.0
                            )

                            # Extract grid size
                            response = response.choices[0].message.content
                            if response == "True" or response == "true":
                                temp_dict1[inst1] = symbolic_info[self.obj_tp][inst1]
                                temp_dict2[inst2] = symbolic_info[self.obj_tp2][inst2]

            ret_symb_info[self.obj_tp] = temp_dict1
            ret_symb_info[self.obj_tp2] = temp_dict2

        elif self.where_tp == "and":
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

                            if inst in left_symb_info[obj_tp]:
                                inleft = True

                            if inst in right_symb_info[obj_tp]:
                                inright = True

                            if inleft or inright:
                                temp_dict[inst] = symbolic_info[obj_tp][inst]
                    
                    ret_symb_info[obj_tp] = temp_dict

        elif self.where_tp == "not":
            return f"!({self.sub_where_clause[0].pretty_str()})"

            true_ret_info = self.sub_where_clause[0].filter(symbolic_info)

            # Compare to ret symb info
            # Remove version that are in true_ret_info
            keep_obj_list = []
            if self.obj_tp in symbolic_info:
                for obj_dict in symbolic_info[self.obj_tp]:
                    in_true = False

                    if self.obj_tp in true_ret_info:
                        for t_obj_dict in true_ret_info[self.obj_tp]:
                            if obj_dict['id'] == t_obj_dict['id']:
                                in_true = True

                    if not in_true:
                        keep_obj_list.append(obj_dict)

            ret_symb_info[self.obj_tp] = keep_obj_list

        return ret_symb_info
