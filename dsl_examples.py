from dsl import WhereClause, Count, Query, Map

# Count the number of cups
program = Count(
    query=Query(
        obj_tp="cups", where_clause=WhereClause(where_tp="true", obj_tp="cups")
    ),
    obj_tp="cups",
)

print(program.pretty_str())


# Find vacant conference room
sub_where1 = WhereClause(
    where_tp="not",
    obj_tp="Conference Room",
    sub_where_clause=[
        WhereClause(
            where_tp="spatial_rel",
            obj_tp="Conference Room",
            obj_tp2="Human",
            spatial_relation="inside",
        )
    ],
)
sub_where2 = WhereClause(
    where_tp="spatial_rel",
    obj_tp="Conference Room",
    obj_tp2="White Board",
    spatial_relation="inside",
)
program = Map(
    obj_tp="Conference Room",
    map_feature="location",
    query=Query(
        obj_tp="Conference Room",
        where_clause=WhereClause(
            where_tp="and",
            obj_tp="Conference Room",
            sub_where_clause=[sub_where1, sub_where2],
        ),
    ),
)


print(sub_where1.pretty_str())
print(sub_where2.pretty_str())

print(program.pretty_str())


# Where is my tallest cup
program = Map(
    obj_tp="Cup",
    map_feature="location",
    query=Query(
        obj_tp="Cup",
        where_clause=WhereClause(where_tp="max", obj_tp="cup", scalar_feature="height"),
    ),
)
print(program.pretty_str())
