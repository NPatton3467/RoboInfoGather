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
    def __init__(self, obj_tp, map_feature, query):
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
            for obj_inst in self.query.result[self.obj_tp]:
                self.result[obj_inst] = obj_inst[self.map_feature]

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

    def execute(self):
        if self.result == None:
            if self.prim_tp == "real":
                self.result = self.prim
            elif self.prim_tp == "op":
                # Execute left and right primitives
                left = 0
                right = 0
                if type(self.prim) in [GetNth, Count, Aggregator]:
                    if self.prim.result == None:
                        self.prim.execute()

                    left = self.prim.result
                else:
                    left = self.prim
                
                if type(self.prim2) in [GetNth, Count, Aggregator]:
                    if self.prim2.result == None:
                        self.prim2.execute()

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
            key = list(self.list.keys())[self.index]
            self.result = self.list[key]

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
            self.result = len(self.query.result[self.obj_tp])

        return self.result


class Aggregator:
    def __init__(self, agg_tp, symbolic_list):
        self.agg_tp = agg_tp
        self.list = symbolic_list

        # Result
        self.result = None

    def pretty_str(self):
        return f"{self.agg_tp}({self.list.pretty_str})"

    def execute(self, symbolic_info):
        # List must have been completed
        if self.list.result == {}:
            self.list.execute(symbolic_info)

        if self.result == None:
            if self.agg_tp == "sum":
                sum = 0
                
                for key in self.list.result:
                    sum += self.list.result[key]

                self.result = sum

            elif self.agg_tp == "avg":
                avg = 0

                for key in self.list.result:
                    avg += self.list.result[key]

                self.result = avg/len(self.list.result)

            elif self.agg_tp == "min":
                min = -1

                for key in self.list.result:
                    if self.list.result[key] < min or min == -1:
                        min = self.list.result[key]

                self.result = min

            elif self.agg_tp == "max":
                max = -1

                for key in self.list:
                    if self.list.result[key] > max or max == -1:
                        max = self.list.result[key]

                self.result = max

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
            self.result = self.where_clause.filter(symbolic_info)

            if len(self.result) > self.limit:
                self.result = self.result[0:self.limit]

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
        ret_symb_info = symbolic_info
        if self.where_tp == "feature_enum":
            temp_list = []
            for obj_dict in ret_symb_info[self.obj_tp]:
                if self.enum_feature in obj_dict and obj_dict[self.enum_feature] == self.enum_param:
                    temp_list.append(obj_dict)

            ret_symb_info[self.obj_tp] = temp_list

        elif self.where_tp == "feature_scalar":
            if self.scalar_comparator == "Lt":
                temp_list = []
                for obj_dict in ret_symb_info[self.obj_tp]:
                    if self.scalar_feature in obj_dict and obj_dict[self.scalar_feature] < self.scalar_param:
                        temp_list.append(obj_dict)

                ret_symb_info[self.obj_tp] = temp_list

            elif self.scalar_comparator == "Leq":
                temp_list = []
                for obj_dict in ret_symb_info[self.obj_tp]:
                    if self.scalar_feature in obj_dict and obj_dict[self.scalar_feature] <= self.scalar_param:
                        temp_list.append(obj_dict)

                ret_symb_info[self.obj_tp] = temp_list
                
            elif self.scalar_comparator == "Eq":
                temp_list = []
                for obj_dict in ret_symb_info[self.obj_tp]:
                    if self.scalar_feature in obj_dict and obj_dict[self.scalar_feature] == self.scalar_param:
                        temp_list.append(obj_dict)

                ret_symb_info[self.obj_tp] = temp_list
                
            elif self.scalar_comparator == "Geq":
                temp_list = []
                for obj_dict in ret_symb_info[self.obj_tp]:
                    if self.scalar_feature in obj_dict and obj_dict[self.scalar_feature] >= self.scalar_param:
                        temp_list.append(obj_dict)

                ret_symb_info[self.obj_tp] = temp_list
                
            elif self.scalar_comparator == "Gt":
                temp_list = []
                for obj_dict in ret_symb_info[self.obj_tp]:
                    if self.scalar_feature in obj_dict and obj_dict[self.scalar_feature] > self.scalar_param:
                        temp_list.append(obj_dict)

                ret_symb_info[self.obj_tp] = temp_list
                

        elif self.where_tp == "max":
            temp_obj = None
            max_val = -1
            for obj_dict in ret_symb_info[self.obj_tp]:
                if self.scalar_feature in obj_dict and (obj_dict[self.scalar_feature] > max_val or max_val == -1):
                    temp_obj = obj_dict
                    max_val = obj_dict[self.scalar_feature]

            ret_symb_info[self.obj_tp] = temp_obj

        elif self.where_tp == "min":
            temp_obj = None
            min_val = -1
            for obj_dict in ret_symb_info[self.obj_tp]:
                if self.scalar_feature in obj_dict and (obj_dict[self.scalar_feature] > min_val or min_val == -1):
                    temp_obj = obj_dict
                    min_val = obj_dict[self.scalar_feature]

            ret_symb_info[self.obj_tp] = temp_obj

        elif self.where_tp == "spatial_rel":
            assert False # What to do here?
            return f"{self.spatial_relation}({self.obj_tp}, {self.obj_tp2})"

        elif self.where_tp == "and":
            ret_symb_info = self.sub_where_clause[0].filter(ret_symb_info)
            ret_symb_info = self.sub_where_clause[1].filter(ret_symb_info)

        elif self.where_tp == "or":
            left_symb_info = self.sub_where_clause[0].filter(ret_symb_info)
            right_symb_info = self.sub_where_clause[1].filter(ret_symb_info)

            # Combine
            temp_ret_info = {}
            for obj_tp in ret_symb_info:
                if obj_tp in left_symb_info or obj_tp in right_symb_info:
                    temp_list = []
                    for obj_dict in ret_symb_info[obj_tp]:
                        inleft = False
                        inright = False

                        for l_obj_dict in left_symb_info[obj_tp]:
                            if obj_dict['id'] == l_obj_dict['id']:
                                inleft = True
                                break

                        for r_obj_dict in right_symb_info[obj_tp]:
                            if obj_dict['id'] == r_obj_dict['id']:
                                inright = True
                                break

                        if inleft or inright:
                            temp_list.append(obj_dict)
                    
                    temp_ret_info[obj_tp] = temp_list

            ret_symb_info = temp_ret_info

        elif self.where_tp == "not":
            return f"!({self.sub_where_clause[0].pretty_str()})"

            true_ret_info = self.sub_where_clause[0].filter(ret_symb_info)

            # Compare to ret symb info
            # Remove version that are in true_ret_info
            keep_obj_list = []
            for obj_dict in ret_symb_info[self.obj_tp]:
                in_true = False

                for t_obj_dict in true_ret_info[self.obj_tp]:
                    if obj_dict['id'] == t_obj_dict['id']:
                        in_true = True

                if not in_true:
                    keep_obj_list.append(obj_dict)

            ret_symb_info[self.obj_tp] = keep_obj_list

        return ret_symb_info
