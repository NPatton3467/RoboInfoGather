from dsl import *

# Count the number of cups
ex_1 = Count(Query('cups', WhereClause('true', 'cups')), 'cups')

print(ex_1.pretty_str())


# Find vacant conference room
sub_where1 = WhereClause('not', 'Conference Room', sub_where_clause=[WhereClause('spatial_rel', 'Conference Room', obj_tp2='Human', spatial_relation='inside')])
sub_where2 = WhereClause('spatial_rel', 'Conference Room', obj_tp2='White Board', spatial_relation='inside')
ex_2 = Map('Conference Room', 'location', Query('Conference Room', WhereClause('and', 'Conference Room', sub_where_clause=[sub_where1, sub_where2])))


print(sub_where1.pretty_str())
print(sub_where2.pretty_str())

print(ex_2.pretty_str())


# Where is my tallest cup 
ex_3 = Map('Cup', 'location', Query('Cup', WhereClause('max', 'cup', scalar_feature='height')))
print(ex_3.pretty_str())
