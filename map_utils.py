import numpy as np
import os
from PIL import Image
#import cv2
import openai
from openai import OpenAI
f = open('/robodata/user_data/npatt/explore-eqa/RoboInfoGather/openaikey.txt', 'r')
openai_api_key = f.read().rstrip('\n')
f.close()

def get_map_params(obj_tp, map_original_dim, map_original_resolution, vol_origin):
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
        model="gpt-4o-mini-2024-07-18",
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

    map_dim = (
        map_original_dim * map_original_resolution / map_resolution
    ).astype(int)

    # Different res for z dimension
    map_dim[2] = int(
        map_original_dim[2] * map_original_resolution / z_resolution
    )

    return {'res' : map_resolution, 'og_res' : map_original_resolution, 'dim' : map_dim, 'og_dim' : map_original_dim, 'z_res' : z_resolution, 'vol_origin': vol_origin}

def world_to_map(xyz, vol_origin, map_resolution, z_resolution, map_dim):
    pts = xyz - vol_origin
    coords = np.round(pts / map_resolution).astype(int)
    coords[2] = np.round(pts[2] / z_resolution).astype(int)
    coords = np.clip(coords, 0, map_dim - 1)
    return coords

def map_to_world(xyz, vol_origin, map_resolution, z_resolution):
    vol_origin = vol_origin.astype(np.float32)
    vox_coords = xyz.astype(np.float32)
    cam_pts = np.empty_like(vox_coords, dtype=np.float32)
    for i in range(3):
        if i < 2:
            cam_pts[i] = vol_origin[i] + (map_resolution * vox_coords[i])
        else:
            cam_pts[i] = vol_origin[i] + (z_resolution * vox_coords[i])
    return cam_pts
