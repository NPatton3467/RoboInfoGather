from RoboInfoGather.program_utils import *

f_found = "The chair is made of metal"
f_real = "Metalic"
comp = "=="

for i in range(10): # Test 10 times
    ret_val = eval_feature_equality(f_found, comp, f_real)
    assert ret_val


f_found = "The chair is made of metal"
f_real = "Metalic"
comp = "!="

for i in range(10): # Test 10 times
    ret_val = eval_feature_equality(f_found, comp, f_real)
    assert not ret_val
