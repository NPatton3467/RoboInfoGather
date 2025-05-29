import numpy as np
import os
from PIL import Image
import cv2
import openai
from openai import OpenAI
f = open('/robodata/user_data/npatt/explore-eqa/RoboInfoGather/openaikey.txt', 'r')
openai_api_key = f.read().rstrip('\n')
f.close()

def get_map_params(obj_tp, map_original_dim, map_original_resolution, vol_origin, config):
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
    obj_map_resolution = float(gresponse)
    
    # Extract vertical grid size
    vresponse = response.choices[0].message.content
    height_idx = vresponse.find('height')
    vresponse = vresponse[height_idx:]
    vresponse = vresponse.lstrip('height = ')
    meters_idx = vresponse.find('meters')
    vresponse = vresponse[:meters_idx].rstrip(' ')
    obj_z_resolution = float(vresponse)

    map_resolution = config['bel_params']['res_to_use']
    z_resolution = config['bel_params']['res_to_use']

    map_dim = (
        map_original_dim * map_original_resolution / map_resolution
    ).astype(int)

    # Different res for z dimension
    map_dim[2] = int(
        map_original_dim[2] * map_original_resolution / z_resolution
    )

    return {'res' : map_resolution, 'og_res' : map_original_resolution, 'dim' : map_dim, 'og_dim' : map_original_dim, 'z_res' : z_resolution, 'vol_origin': vol_origin, 'obj_map_res': obj_map_resolution, 'obj_z_res': obj_z_resolution}

# Convert world coordinates to other map coordinates (e.g. belief)
def world_to_map(xyz, vol_origin, map_resolution, z_resolution, map_dim):
    pts = xyz - vol_origin
    coords = np.round(pts / map_resolution).astype(int)
    coords[2] = np.round(pts[2] / z_resolution).astype(int)
    coords = np.clip(coords, 0, map_dim - 1)
    return coords

# Convert map coordinates (e.g. belief) to world coordinates
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

def get_trav_map(maps_path, floor, resolution, og_resolution):
        """
        Loads the traversability map
        """

        if not os.path.exists(maps_path):
            assert False
            log.warning("trav map does not exist: {}".format(maps_path))
            return

        map_size = None
        trav_map = np.array(Image.open(os.path.join(maps_path, "floor_trav_no_obj_{}.png".format(floor))))

        # If we do not initialize the original size of the traversability map, we obtain it from the image
        # Then, we compute the final map size as the factor of scaling (default_resolution/resolution) times the
        # original map size
        height, width = trav_map.shape
        map_default_resolution = og_resolution
        map_size = int(
            height * map_default_resolution / resolution
        )

        # We resize the traversability map to the new size computed before
        trav_map = cv2.resize(trav_map, (map_size, map_size))

        # We make the pixels of the image to be either 0 or 255
        trav_map[trav_map < 255] = 0

        return trav_map
