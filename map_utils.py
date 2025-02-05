import numpy as np
import os
from PIL import Image
#import cv2
import openai
from openai import OpenAI
f = open('/robodata/user_data/npatt/explore-eqa/RoboInfoGather/openaikey.txt', 'r')
openai_api_key = f.read().rstrip('\n')
f.close()

def get_map_params(obj_tp, map_original_size, map_original_resolution):
    """
    Calculates parameters needed for mapping form world to Belief coordinates using LLM and previous params

    :param: (obj_tp) Object type used to query LLM for new map resolution
    :param: (map_original_size) The original size of the the traversable map
    :param: (map_original_resolution) The original resolution of the traversable map

    :returns: a dict {'res' : map_resolution, 'og_res' : map_original_resolution, 'size' : map_size, 'og_size' : map_original_size}
    """

    # Query LLM for map resolution
    f = open('./RoboInfoGather/res_pre_prompt.txt', 'r')
    pre_prompt = f.read()
    f.close()
    prompt = pre_prompt + str(obj_tp) + '\n```\nOutput:\n```\n'

    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"{prompt}"}],
        stream=False,
        temperature=0.0
    )

    # Extract grid size
    gresponse = response.choices[0].message.content
    grid_size_idx = gresponse.find('width')
    gresponse = gresponse[grid_size_idx:]
    gresponse = gresponse.lstrip('width = ')
    meters_idx = gresponse.find('meters')
    gresponse = gresponse[:meters_idx].rstrip(' ')
    map_resolution = float(gresponse)
    
    # Extract vertical grid size
    vresponse = response.choices[0].message.content
    height_idx = vresponse.find('height')
    vresponse = vresponse[height_idx:]
    vresponse = vresponse.lstrip('height = ')
    meters_idx = vresponse.find('meters')
    vresponse = vresponse[:meters_idx].rstrip(' ')
    z_resolution = float(vresponse)

    map_size = map_size = int(
        map_original_size * map_original_resolution / map_resolution
    )

    return {'res' : map_resolution, 'og_res' : map_original_resolution, 'size' : map_size, 'og_size' : map_original_size, 'z_res' : z_resolution}

def world_to_map(xy, map_resolution, map_size):
    return np.flip((np.array(xy) / map_resolution + map_size / 2.0)).astype(np.int)

def map_to_world(xy, map_resolution, map_size):
    axis = 0 if len(xy.shape) == 1 else 1
    return np.flip((xy - map_size / 2.0) * map_resolution, axis=axis)
