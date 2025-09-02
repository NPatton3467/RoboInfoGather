import json
import numpy as np
import cv2
from RoboInfoGather.map_utils import *
#from groundingdino.util.inference import predict
#import groundingdino.datasets.transforms as T
from RoboInfoGather.MCTS_planner import Loc

from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision.transforms.functional as TF

from scipy.spatial.transform import Rotation as R
import skimage.measure

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt 
import glob

import openai
from openai import OpenAI
f = open('/robodata/user_data/npatt/explore-eqa/RoboInfoGather/openaikey.txt', 'r')
openai_api_key = f.read().rstrip('\n')
f.close()

import base64
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# For structured GPT output
from pydantic import BaseModel
import instructor
from typing import Literal

# For interactive observations
import tkinter as tk
from RoboInfoGather.interactive_pixel_selector import *

# Two clases below are provided to instructor call
# to give structure to GPT output
class Feature(BaseModel):
    object_type: str
    feature_type: str
    feature_val: str

class Exists(BaseModel):
    exists: Literal['Yes','No']

class EquivalentClass(BaseModel):
    intended_object_type: str
    instance_object_type: str
    equivalent: bool

# Function to encode the image for GPT
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function for getting point cloud from depth
def get_new_points(depth_image, camera_pos, camera_pose, cam_int_mat):
    pts = []
    for p_x in range(depth_image.shape[1]):
        for p_y in range(depth_image.shape[0]):
            cur_depth = depth_image[p_y, p_x]
            world_coords = get_world_coords_from_depth(p_x, p_y, cur_depth, camera_pos, camera_pose, cam_int_mat)
            pts.append(world_coords)

    return pts

def get_bbox_3d_corners(bbox):
    """Return transformed points in the following order: [LDB, RDB, LUB, RUB, LDF, RDF, LUF, RUF]
    where R=Right, L=Left, D=Down, U=Up, B=Back, F=Front and LR: x-axis, UD: y-axis, FB: z-axis.

    Args:
        extents (numpy.ndarray): A structured numpy array containing the fields: [`x_min`, `y_min`,
            `x_max`, `y_max`, `transform`.

    Returns:
        (numpy.ndarray): Transformed corner coordinates with shape `(N, 8, 3)`.
    """

    extents = {
            "x_min": bbox[1],
            "y_min": bbox[2],
            "z_min": bbox[3],
            "x_max": bbox[4],
            "y_max": bbox[5],
            "z_max": bbox[6],
            "transform": bbox[7]
    }

    tfs = extents["transform"]
    rdb = np.matmul([extents["x_max"], extents["y_min"], extents["z_min"], 1], tfs)[:3]
    ldb = np.matmul([extents["x_min"], extents["y_min"], extents["z_min"], 1], tfs)[:3]
    lub = np.matmul([extents["x_min"], extents["y_max"], extents["z_min"], 1], tfs)[:3]
    rub = np.matmul([extents["x_max"], extents["y_max"], extents["z_min"], 1], tfs)[:3]
    ldf = np.matmul([extents["x_min"], extents["y_min"], extents["z_max"], 1], tfs)[:3]
    rdf = np.matmul([extents["x_max"], extents["y_min"], extents["z_max"], 1], tfs)[:3]
    luf = np.matmul([extents["x_min"], extents["y_max"], extents["z_max"], 1], tfs)[:3]
    ruf = np.matmul([extents["x_max"], extents["y_max"], extents["z_max"], 1], tfs)[:3]

    corners = np.stack((ldb, rdb, lub, rub, ldf, rdf, luf, ruf), 0)
    #corners_homo = np.pad(corners, ((0, 0), (0, 1)), constant_values=1.0)
    #print(corners.shape)
    #print(corners_homo.shape)

    #return np.einsum("jki,ikl->ijl", corners_homo, tfs)[..., :3]

    x_min = None
    x_max = None
    y_min = None
    y_max = None
    z_min = None
    z_max = None

    for coord in corners:
        if x_min == None or coord[0] < x_min:
            x_min = coord[0]
        if y_min == None or coord[1] < y_min:
            y_min = coord[1]
        if z_min == None or coord[2] < z_min:
            z_min = coord[2]
        
        if x_max == None or coord[0] > x_max:
            x_max = coord[0]
        if y_max == None or coord[1] > y_max:
            y_max = coord[1]
        if z_max == None or coord[2] > z_max:
            z_max = coord[2]

    return x_min, y_min, z_min, x_max, y_max, z_max


# Get new point in real coords based on robot position
def get_new_node(current_loc, dist, angle, belief):
    # In robot frame: robot direction is X-axis.

    # Find the X,Y locations of the point in robot frame 
    # from distance and angle
    new_x = np.cos(angle) * dist
    new_y = np.sin(angle) * dist

    # Do a rotation based on robot theta to get delta x,y in real coords
    delta_x = new_x * np.cos(current_loc.theta) - new_y * np.sin(current_loc.theta)
    delta_y = new_x * np.sin(current_loc.theta) + new_y * np.cos(current_loc.theta)

    # Get actual x and y based off of current loc
    real_x = current_loc.x + delta_x
    real_y = current_loc.y + delta_y

    # Get Map xy
    return real_x, real_y

# Use camera params to get real world coordinates from (x,y) pixel and depth image
def get_world_coords_from_depth(x, y, depth, camera_pos, camera_pose, camera_intrinsic_mat):

    """
    Use the camera params and depth image to get simulator map frame coordinates from pixel coordinates

    Inputs:
        x:                      The x coordinate of the pixel to be evaluated (int)
        y:                      The y coordinate of the pixel to be evaluated (int)
        depth:                  The current DEPTH image observation
        camera_pos:             The current camera position in the simulator map frame
        camera_pose:            The current camera rotation + translation matrix in the simulator map frame
        camera_intrinsic_mat:   The camera intrinsic matrix used to perform the projection

    Outputs:
        world_coords:           The simulator map frame coordinates of the pixel (x,y) and depth value (z).
    """

    cx = camera_intrinsic_mat[0][2]
    cy = camera_intrinsic_mat[1][2]
    fx = camera_intrinsic_mat[0][0]
    fy = camera_intrinsic_mat[1][1]
    
    camera_coords_z = depth
    camera_coords_x = (x-cx)*camera_coords_z/fx
    camera_coords_y = (y-cy)*camera_coords_z/fy

    # Translate 3D coordinates to global frame
    c_coord = np.array([camera_coords_x, camera_coords_y, camera_coords_z, 1])

    # Coordinate transform
    c_coord2 = np.array([c_coord[2], -c_coord[0], -c_coord[1], 1])

    # Rotate to yaw 

    world_coords = np.matmul(camera_pose, c_coord2)
    return world_coords[0:3]

# Iterate throught pixels in depth image
# Get depth value as maximum depth
# Get all the free space from the camera to that point in space
# Return free space in FOV based on this depth image
def get_fov_from_depth_image(camera_pos, camera_pose, raw_depth_image, voxel_preds, resolution, z_res, dim, vol_origin, config, cam_int_mat):

    """
    Use the current depth image to get a field of view in the belief map frame (aka the voxel frame)

    Inputs:
        camera_pos:             The current camera position in the simulator map frame
        camera_pose:            The current camera rotation + translation matrix in the simulator map frame
        raw_depth_image:        The current DEPTH image observation
        voxel_preds:            An empty array, with the same shape as the belief being considered
        resolution:             The voxel resolution of the belief
        z_res:                  The voxel resolution of the belief's z-axis
        dim:                    The dimensions of the belief being considered
        vol_origin:             The offset of the belief's (0,0,0) coordinate w.r.t. the simulator map frame
        config:                 The current task configuration data
        cam_int_mat:            The camera intrinsic matrix used for coordinate transforms between
                                    pixel coordinate frames and simulator map frame

    Outputs:
        voxel_preds:            The input 'voxel_preds' array with all voxels within the field of view
                                    updated if they are predicted to be free space
    """

    # Max pool to decrease image size
    depth_image = skimage.measure.block_reduce(raw_depth_image, (16,16), np.min)
    print("Got Depth Image... Shape: ", depth_image.shape)

    # Set to zero up to obstacle
    for p_x in range(depth_image.shape[1]):
        for p_y in range(depth_image.shape[0]):
            #print(p_x, p_y)
            max_depth = depth_image[p_y, p_x]
            if max_depth >= np.max(dim) * resolution:
                continue

            # Resize p_x, and p_y to original location in image for real world coordinate mapping
            x = int(p_x / depth_image.shape[1] * raw_depth_image.shape[1])
            y = int(p_y / depth_image.shape[0] * raw_depth_image.shape[0])

            cur_depth = 0
            while cur_depth < max_depth:
                world_coords = get_world_coords_from_depth(x, y, cur_depth, camera_pos, camera_pose, cam_int_mat)

                # Set voxel pred location to 0 here
                v_xyz = world_to_map(world_coords, vol_origin, resolution, z_res, dim)
                if not np.isnan(world_coords[2]):
                    voxel_preds[v_xyz[0], v_xyz[1], v_xyz[2]] = config['observation_calc_params']['prob_occ_given_obs_free']

                cur_depth += (resolution / 2)


    print("Done Depth Loop")
    
    return voxel_preds
                
# Get FOV based only on known obstacle map and camera orientation
def get_fov(current_location, config, camera_params, obstacle_map, belief, debug_print=True):
    min_angle = camera_params['min_angle']
    max_angle = camera_params['max_angle']
    min_v_dist = camera_params['min_visual_distance']
    max_v_dist = camera_params['max_visual_distance']

    angle_delta = config['rf_params']['angle_delta']
    dist_delta = min(0.15, obstacle_map._voxel_size * 0.8)

    angle = min_angle

    # Use inflated and resized obstacle map to not see through walls?
    inflated_resized_obstacle_map = obstacle_map._tsdf_vol_cpu

    fov = []
    obstacles = []
    while angle < max_angle:
        dist = 0
        if debug_print:
            print(len(fov))
        while dist < max_v_dist:
            # Get node that corresponds to angle and dist (relative to robot)
            x, y = get_new_node(current_location, dist, angle, belief)

            # Skip if already added
            if (x, y) in fov:
                assert False #???
                dist += dist_delta
                continue

            # Check that x,y are within map bounds and not occluded 
            o_size_x = inflated_resized_obstacle_map.shape[0]
            o_size_y = inflated_resized_obstacle_map.shape[1]
            o_xy = obstacle_map.world2vox(np.array([x, y, 0]))
            if o_xy[0] not in range(0, o_size_x) or o_xy[1] not in range(0, o_size_y):
                break
            

            # Check if new location would cause a collision
            height_voxel = int(0.4 / obstacle_map._voxel_size) + obstacle_map.min_height_voxel
            unoccupied = np.logical_and(
                obstacle_map._tsdf_vol_cpu[o_xy[0], o_xy[1], height_voxel] > 0, obstacle_map._tsdf_vol_cpu[o_xy[0], o_xy[1], 0] < 0
            )

            if unoccupied:
                obstacles.append((x,y))
                break

            if dist > min_v_dist:
                fov.append((x,y))

            # Increment distance
            dist += dist_delta

        # Increment angle
        angle += angle_delta

    return fov, obstacles

# Query VLM to see if an instance of obj_tp exists in img
def instance_exists(img, obj_tp):

    """
    Check whether an instance of a given object type exists within the image

    Inputs:
        img:        The current RGB image observation
        obj_tp:     The object type to check for instances of

    Outputs:
        inst_e:     A boolean representing whether or not an instance of 'obj_tp' exists
                        in 'img'
    """

    prompt = f"Is it fairly likely that there is an instance of \'{obj_tp}\' in this image? Please respond with only \'Yes\' or \'No\'."

    buffered = BytesIO()
    print("Image Size: ", img.size)
    img.save(buffered, format="JPEG")
    cur_img_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    cur_message = {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url":{"url" : f"data:image/png;base64,{cur_img_encoded}"}
                        }
                    ]
            }
    messages = [
        cur_message
    ]
    
    # Query VLM
    client = instructor.from_openai(OpenAI(api_key=openai_api_key), mode=instructor.Mode.MD_JSON)
    response = client.chat.completions.create(
      model="gpt-4o-mini-2024-07-18",
      response_model=Exists,
      messages=messages,
      max_tokens=300,
    )

    inst_e = (response.exists == 'Yes')

    return inst_e

# Query MOLMO to get pixel coordinates of object instances in img
def get_pixel_coords_molmo(molmo_tools, obj_tp, img):

    """
    Get the pixel locations of any object instances of a given type

    Inputs:
        molmo_tools:            The molmo model and processor used for getting the pixel values
        obj_tp:                 The object type to get pixel locations of
        img:                    The current RGB image observation

    Outputs:
        coords: `               The pixel values cooresponding to instances of 'obj_tp' in 'img'
                                    as predicted by the molmo model. Each instance of 'obj_tp' should
                                    correspond to a single pixel coordinate.
    """

    # Down Sample image if too big
    down_sampled_img = img.resize((500, 500), Image.Resampling.LANCZOS)
    print("Image Size for MOLMO: ", down_sampled_img.size)

    coords = []
    molmo_model = molmo_tools['model']
    processor = molmo_tools['processor']
    # Process with MOLMO
    with open('./RoboInfoGather/molmo_preprompt.txt', 'r') as f:
        pre_prompt = f.read()

    prompt = pre_prompt + f"Now, please provide the pixel coordinates corresponding to the centroid of any {obj_tp}(s) that are in this image.\n"

    with torch.autocast("cuda", enabled=True, dtype=torch.float16):
        inputs = processor.process(
            images=[down_sampled_img],
            text= prompt
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(molmo_model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = molmo_model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Get pixel coords
    temp_gen_text = generated_text
    while temp_gen_text.find('(') >= 0:
        idx = temp_gen_text.find('(') + 1
        temp_gen_text = temp_gen_text[idx:]
       
        try:
            xy = temp_gen_text.split(',')
            x = float(xy[0])
            y = xy[1]
            y_end_idx = y.find(')')
            y = float(y[:y_end_idx].lstrip(' '))
            
            if x >= 0 and x < 100 and y >= 0 and y < 100:
                x = int(x * img.shape[1])
                y = int(y * img.shape[0])
                coords.append((x,y))
        except Exception as e:
            print(f"Failed to create pixel {xy} from output")

    print("MOLMO GEN COORDS: ", coords)

    return coords

# Template to create messages
class FeatureMessage():
    def __init__(self, obj_tp, feature, expected_val=''):
        self.obj_tp = obj_tp
        self.feature = feature
        self.expected_val = expected_val

    def make_msg(self):
        preamble = "You are provided with an object type, and a corresponding feature to evaluate within the image. Please only provide the value of the feature that corresponds to the instance marked with a white circle with the letter `A` inside the white circle."
        msg = preamble + f"\nObject Type: {self.obj_tp}\nFeature to evaluate: {self.feature}\n\nValue: {self.expected_val}"
        return msg


def get_pre_prompt_msgs():
    # Path to your image
    image_path = "./RoboInfoGather/feature_pre_prompt_figs/1_black_chair_2_brown_chairs__black.png"

    # Getting the Base64 string
    black_chair_image = encode_image(image_path)
    fm = FeatureMessage(obj_tp='Chair', feature='Colour', expected_val='Black')
    message_1 = {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": "Example 1: " + fm.make_msg()
                        },
                        {
                            "type": "image_url",
                            "image_url":{"url" : f"data:image/png;base64,{black_chair_image}"}
                        }
                    ]
            }

    # Path to your image
    image_path = "./RoboInfoGather/feature_pre_prompt_figs/1_black_chair_2_brown_chairs__brown.png"

    # Getting the Base64 string
    brown_chair_image = encode_image(image_path)
    fm = FeatureMessage(obj_tp='Chair', feature='Colour', expected_val='Brown')
    message_2 = {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": "Example 2: " + fm.make_msg()
                        },
                        {
                            "type": "image_url",
                            "image_url":{"url" : f"data:image/png;base64,{brown_chair_image}"}
                        }
                    ]
            }
    
    # Path to your image
    image_path = "./RoboInfoGather/feature_pre_prompt_figs/plant_near_couch.png"

    # Getting the Base64 string
    plant_image = encode_image(image_path)
    fm = FeatureMessage(obj_tp='Plant', feature='Near Couch', expected_val='True')
    message_3 = {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": "Example 3: " + fm.make_msg()
                        },
                        {
                            "type": "image_url",
                            "image_url":{"url" : f"data:image/png;base64,{plant_image}"}
                        }
                    ]
            }
    
    # Path to your image
    image_path = "./RoboInfoGather/feature_pre_prompt_figs/tv_off.png"

    # Getting the Base64 string
    tv_image = encode_image(image_path)
    fm = FeatureMessage(obj_tp='TV', feature='Power Status', expected_val='Off')
    message_4 = {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": "Example 4: " + fm.make_msg()
                        },
                        {
                            "type": "image_url",
                            "image_url":{"url" : f"data:image/png;base64,{tv_image}"}
                        }
                    ]
            }
    
    # Path to your image
    image_path = "./RoboInfoGather/feature_pre_prompt_figs/light_above_table.png"

    # Getting the Base64 string
    light_image = encode_image(image_path)
    fm = FeatureMessage(obj_tp='Light', feature='Below Table', expected_val='False')
    message_5 = {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": "Example 5: " + fm.make_msg()
                        },
                        {
                            "type": "image_url",
                            "image_url":{"url" : f"data:image/png;base64,{light_image}"}
                        }
                    ]
            }
    
    # Path to your image
    image_path = "./RoboInfoGather/feature_pre_prompt_figs/sink_porcelain.png"

    # Getting the Base64 string
    sink_image = encode_image(image_path)
    fm = FeatureMessage(obj_tp='Sink', feature='Material', expected_val='Porcelain')
    message_6 = {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": "Example 6: " + fm.make_msg()
                        },
                        {
                            "type": "image_url",
                            "image_url":{"url" : f"data:image/png;base64,{sink_image}"}
                        }
                    ]
            }
    
    # Path to your image
    image_path = "./RoboInfoGather/feature_pre_prompt_figs/fridge_not_near_microwave.png"

    # Getting the Base64 string
    fridge_image = encode_image(image_path)
    fm = FeatureMessage(obj_tp='Fridge', feature='Near Microwave', expected_val='False')
    message_7 = {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": "Example 7: " + fm.make_msg()
                        },
                        {
                            "type": "image_url",
                            "image_url":{"url" : f"data:image/png;base64,{fridge_image}"}
                        }
                    ]
            }

    messages = [
        message_1,
        message_2,
        message_3,
        message_4,
        message_5,
        message_6,
        message_7
    ]

    return messages

def get_feature_vals_cropped(cropped_img, coord, obj_tp, feature, cfg):

    """
    Get the value of a given feature corresponding to an object instance of a given type

    Inputs:
        cropped_img:            The current RGB image observation, cropped around the object instance
        coord:                  The pixel coordinate of the instance
        obj_tp:                 The object type of the current instance
        feature:                The feature to be evalutated

    Outputs:
        responsed.feature_val:  The predicted value of 'feature' of the current instance of 'obj_tp'
    """

    img = Image.fromarray(cropped_img).convert('RGB')
    messages = []

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    cur_img_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    preamble = "You are provided with an object type, and a corresponding feature to evaluate within the image. Please only provide the value of the feature that corresponds to the instance witin the cropped image."
    msg = preamble + f"\nObject Type: {obj_tp}\nFeature to evaluate: {feature}\n\nValue: "
    cur_message = {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": msg,
                        },
                        {
                            "type": "image_url",
                            "image_url":{"url" : f"data:image/png;base64,{cur_img_encoded}"}
                        }
                    ]
            }

    messages.append(cur_message)
    
    # Query VLM
    client = instructor.from_openai(OpenAI(api_key=openai_api_key), mode=instructor.Mode.MD_JSON)
    response = client.chat.completions.create(
      model="gpt-4o-mini-2024-07-18",
      response_model=Feature,
      messages=messages,
      max_tokens=300,
    )

    print("Object Type: ", obj_tp, " Feature: ", feature)
    print("Message Text: ", cur_message['content'][0]['text'])
    print("Response: ", response)

    return response.feature_val

# Query VLM to get feature values of the object instance in the cropped image
def get_feature_vals(cropped_img, coord, obj_tp, feature, cfg):

    """
    Get the value of a given feature corresponding to an object instance of a given type

    Inputs:
        cropped_img:            The current RGB image observation, cropped around the object instance
        coord:                  The pixel coordinate of the instance
        obj_tp:                 The object type of the current instance
        feature:                The feature to be evalutated

    Outputs:
        responsed.feature_val:  The predicted value of 'feature' of the current instance of 'obj_tp'
    """

    img = Image.fromarray(cropped_img).convert('RGB')
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        (
            coord[0] - cfg.visual_prompt.circle_radius,
            coord[1] - cfg.visual_prompt.circle_radius,
            coord[0] + cfg.visual_prompt.circle_radius,
            coord[1] + cfg.visual_prompt.circle_radius,
        ),
        fill=(200, 200, 200, 255),
        outline=(0, 0, 0, 255),
        width=3,
    )

    draw.text(
        tuple(coord.astype(int).tolist()),
        'A',
        fill=(0, 0, 0, 255),
        anchor="mm",
        font_size=15,
    )

    #import matplotlib as mpl
    #mpl.use('TkAgg')
    #from matplotlib import pyplot as plt 
    #plt.imshow(img)
    print("Showing labeled image")
    #plt.show()
    #import matplotlib as mpl
    #mpl.use('Agg')
    #from matplotlib import pyplot as plt 

    ##################################################
    # SET UP PROMPT WITH PREVIOUS IMAGES AS EXAMPLES #
    ##################################################

    # TEMP -- No preprompt?
    #messages = get_pre_prompt_msgs()
    messages = []
    # END TEMP

    
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    cur_img_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    fm = FeatureMessage(obj_tp=obj_tp, feature=feature)
    cur_message = {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": "Given the above examples, please provide an answer for this final message and image: " + fm.make_msg()
                        },
                        {
                            "type": "image_url",
                            "image_url":{"url" : f"data:image/png;base64,{cur_img_encoded}"}
                        }
                    ]
            }

    messages.append(cur_message)
    
    # Query VLM
    client = instructor.from_openai(OpenAI(api_key=openai_api_key), mode=instructor.Mode.MD_JSON)
    response = client.chat.completions.create(
      model="gpt-4o-mini-2024-07-18",
      response_model=Feature,
      messages=messages,
      max_tokens=300,
    )

    print("Object Type: ", obj_tp, " Feature: ", feature)
    print("Message Text: ", cur_message['content'][0]['text'])
    print("Response: ", response)

    return response.feature_val

def equivalent_to_obj_type(instance_type, obj_tp):
    """
    Use LLM to determin if the two strings are equivalent
    """

    prompt = f"Is the object instance of type {instance_type} equivalent to the object type {obj_tp}?"

    cur_message = {
        "role": "user",
        "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ]
    }
    messages = [
        cur_message
    ]
    
    # Query VLM
    client = instructor.from_openai(OpenAI(api_key=openai_api_key), mode=instructor.Mode.MD_JSON)
    response = client.chat.completions.create(
      model="gpt-4o-mini-2024-07-18",
      response_model=EquivalentClass,
      messages=messages,
      max_tokens=300,
    )

    print("\nPrompt\n", prompt)
    print("\nResponse\n", response)

    return response.equivalent

def get_approx_centroid(seg_inst, inst_id):
    """
    Get the approximate middle pixel of this semantic instance
    """

    candidate_centroids = np.argwhere(np.array(seg_inst.detach().cpu()) == inst_id)

    max_x = np.max(candidate_centroids[:,0])
    min_x = np.min(candidate_centroids[:,0])
    max_y = np.max(candidate_centroids[:,1])
    min_y = np.min(candidate_centroids[:,1])

    center_of_bbox = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2])

    # Get candidate pixels closest to center of bbox
    best_pix = candidate_centroids[0]
    best_dist = np.sqrt((best_pix[0] - center_of_bbox[0])**2 + (best_pix[1] - center_of_bbox[1])**2)
    for pix in candidate_centroids:
        dist = np.sqrt((pix[0] - center_of_bbox[0])**2 + (pix[1] - center_of_bbox[1])**2)

        if dist < best_dist:
            best_dist = dist
            best_pix = pix

    p_x = best_pix[1]
    p_y = best_pix[0]

    ret_pix = np.array([p_x, p_y])

    assert seg_inst[ret_pix[1]][ret_pix[0]] == inst_id

    return ret_pix


def get_pixel_coords_from_sim_data(obj_tp, obs):
    """
    Used for debugging. Prompts user to select coordinates in the image based on obj_tp

    Inputs:
        obj_tp:         The object type to look for in the image
        obs:            The observation from the simulator -- included semantic segmentation

    Ouputs:
        coords:         A list of pixel coordinates cooresponding to the user's selection
    """

    # Get the instance IDs in the observation
    unique_semantic_inst_ids = np.unique(obs['seg_inst'].detach().cpu())
    
    # Get the semantic names
    class_names = []
    for inst_id in unique_semantic_inst_ids:
        class_id = np.max(np.where(obs['seg_inst'].detach().cpu() == inst_id, obs['seg_sem'].detach().cpu(), 0))

        class_names.append(obs['info']['obs_info']['rob']['rob:eyes:Camera:0']['seg_semantic'][class_id])

    # Get pixel values for approximate centroid of each instance
    pixels = []
    for i in range(len(unique_semantic_inst_ids)):
        if class_names[i] != "unlabelled" and equivalent_to_obj_type(class_names[i], obj_tp):
            # Check if enough visible pixels
            if np.sum(np.where(obs['seg_inst'].detach().cpu() == inst_id, 1, 0)) > 5000:
                pixels.append(get_approx_centroid(obs['seg_inst'], unique_semantic_inst_ids[i]))

    return pixels


def get_perfect_perception_coords(obj_tp, img):
    """
    Used for debugging. Prompts user to select coordinates in the image based on obj_tp

    Inputs:
        obj_tp:         The object type to look for in the image
        img:            The image to select coordinates in 

    Ouputs:
        coords:         A list of pixel coordinates cooresponding to the user's selection
    """

    # Print object type so user knows
    print(f"Please select all centroids for instances of {obj_tp}")

    root = tk.Tk()
    img = np.copy(img)
    app = PixelSelector(root, img)
    root.mainloop()

    pixels = np.array(app.pixels)

    # Clean up the gui
    root.destroy()
    app.shutdown()
    del(app)
    del(root)

    return pixels

def get_3d_bounding_boxes(obs, p_x, p_y, cur_real_world_coord):
    # Get 3D Bounding Boxes
    seg_id = obs['seg_sem'][p_y, p_x].detach().cpu().item()

    print(seg_id)

    cur_extents = None
    best_dist = -1

    # For 3D bounding box transforms
    for bbox in obs['bbox_3d']:
        if bbox[0] == seg_id:
            x_min, y_min, z_min, x_max, y_max, z_max = get_bbox_3d_corners(bbox)
            if (x_min <= cur_real_world_coord[0] and # x_min
                    y_min <= cur_real_world_coord[1] and # y_min
                    z_min <= cur_real_world_coord[2] and # z_min
                    x_max >= cur_real_world_coord[0] and # x_max
                    y_max >= cur_real_world_coord[1] and # y_max
                    z_max >= cur_real_world_coord[2]): # z_max

                cur_extents = {
                    'x_min': x_min,
                    'y_min': y_min,
                    'z_min': z_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'z_max': z_max
                }

                break
            else: # Keep track of closest
                x_c = (x_min + x_max) / 2
                y_c = (y_min + y_max) / 2
                z_c = (z_min + z_max) / 2

                dist = np.sqrt(
                        (x_c - cur_real_world_coord[0]) ** 2 \
                        + (y_c - cur_real_world_coord[1]) ** 2 \
                        + (z_c - cur_real_world_coord[2]) ** 2)

                if dist < best_dist or best_dist == -1:
                    best_dist = dist
                    cur_extents = {
                        'x_min': x_min,
                        'y_min': y_min,
                        'z_min': z_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'z_max': z_max
                    }

    
    #assert cur_extents != None
    # TEMP THIS IF SHOULDN"T GET EXPLORED BUT WANT TO RUN END TO END
    #if cur_extents == None:
    #    plt.plot([4,5,6])
    #    plt.show()
    #    assert False
    # END TEMP

    return cur_extents

def get_real_world_from_bbox(obs, p_x, p_y, cur_real_world_coord):
    cur_extents = get_3d_bounding_boxes(obs, p_x, p_y, cur_real_world_coord)

    if cur_extents == None:
        inst_id = obs['seg_inst'][p_y,p_x].detach().cpu()
        class_id = np.max(np.where(obs['seg_inst'].detach().cpu() == inst_id, obs['seg_sem'].detach().cpu(), 0))

        print("Class Name: ", obs['info']['obs_info']['rob']['rob:eyes:Camera:0']['seg_semantic'][class_id])
        return None

    x_min = cur_extents['x_min']
    x_max = cur_extents['x_max']
    y_min = cur_extents['y_min']
    y_max = cur_extents['y_max']
    z_min = cur_extents['z_min']
    z_max = cur_extents['z_max']

    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2
    z_c = (z_min + z_max) / 2

    return np.array([x_c, y_c, z_c])


# Use main object detection from image method
def obj_detection_molmo(vlm, molmo_tools, cam_int_mat, obj_tp, obs, config, camera_pos, camera_pose, feature=None, old_pix_coords=None):

    """
    Use molmo to detect pixel values of object instances within image

    Inputs:
        vlm:                Optional VLM (prismatic) which could be used for object detection
        molmo_tools:        Molmo model and processor for object detection. Used for finding pixel values
                                of instances within the image
        cam_int_mat:        Camera intrinsic matrix used to project (pixel coordinate frame <-> 
                                simulator map frame)
        obj_tp:             The object type to locate instances of within the image
        rgb_img:            The current RGB image observation
        depth_img:          The current DEPTH image observation
        config:             The current task configuration data
        camera_pos:         The position of the camera in the simulator map frame
        camera_pose:        The rotation + translation matrix of the camera in the simulator map frame
        feature:            (Optional) a string reperesnting a "feature" to predict about the object of type
                                'obj_tp'
        old_pix_coords:     Pixel coords from previous object detection -- for feature evaluation
    Outputs:
        coords:             A list of coordinates in the pixel coordinate frame representing any 
                                instance of 'obj_tp' found within the current observation
        real_world_coords:  Same as 'coords' but projected into the simulator map frame
        feature_ret_vals:   List of pairs of (voxel-coordinate, feature-value) where voxel-coordinate
                                is in the belief map frame and feature value is the predicted value
                                of 'feature' at that voxel
        logits:             Confidence about the prediction at each of these coordinates
    """
    rgb_img = obs['rgb']
    depth_img = obs['depth']
    img = np.copy(rgb_img)
    #Image should be torch tensor
    img = Image.fromarray(img).convert('RGB')

    if old_pix_coords is None:
        coords = []
        # Use perfect perception if debugging with that is set in config
        if config['use_perfect_perception']:
            if config['use_sim_data_for_perception']:
                coords = get_pixel_coords_from_sim_data(obj_tp, obs) 
            else:
                assert False # Need to make sure pixel values align for feature detection here too
                coords = get_perfect_perception_coords(obj_tp, np.copy(rgb_img))

        else:
            # Check if instance exists (to help with MOLMO false positives)
            # If instance exists, get coordinates from MOLMO
            if instance_exists(img, obj_tp):
                coords = get_pixel_coords_molmo(molmo_tools, obj_tp, img)
    else:
        coords = old_pix_coords

    # With all the pixel coordinates get the real world coordinates
    # and any feature values (if feature is not None)
    feature_ret_vals = [] 
    real_world_coords = []
    ret_pix_coords = []
    for coord in coords:
        cropped_img = np.copy(rgb_img)
        p_x = coord[0]
        p_y = coord[1]

        cur_depth = depth_img[p_y, p_x]

        cur_real_world_coord = get_world_coords_from_depth(p_x, p_y, cur_depth, camera_pos, camera_pose, cam_int_mat)
        if config['use_sim_data_for_perception']:
            cur_real_world_coord = get_real_world_from_bbox(obs, p_x, p_y, cur_real_world_coord)
            if cur_real_world_coord is not None:
                real_world_coords.append(cur_real_world_coord)
                ret_pix_coords.append(coord)
        else:
            real_world_coords.append(cur_real_world_coord)
        if feature != None:
            if feature == 'bbox_3d':
                cur_extents = get_3d_bounding_boxes(obs, p_x, p_y, cur_real_world_coord)
                feature_ret_vals.append(cur_extents)
            else:
                if config['use_sim_data_for_perception']:
                    inst_id = int(obs['seg_inst'][p_y, p_x].detach().cpu())
                    pix_vals = np.array(np.argwhere(obs['seg_inst'].detach().cpu() == inst_id))
                    x_min = np.min(pix_vals[1,:])
                    x_max = np.max(pix_vals[1,:])
                    y_min = np.min(pix_vals[0,:])
                    y_max = np.max(pix_vals[0,:])

                    feature_ret_vals.append(get_feature_vals_cropped(np.copy(cropped_img[y_min:y_max, x_min:x_max]), coord, obj_tp, feature, config))
                else:
                    feature_ret_vals.append(get_feature_vals(cropped_img, coord, obj_tp, feature, config))

    print("Real World Coords Selected: ", real_world_coords)

    # Return the score as well
    logits = []
    for i in range(len(ret_pix_coords)):
        logits.append(0.9)

    return ret_pix_coords, real_world_coords, feature_ret_vals, logits

# Main observation function
# First get all of the free space predictions from the depth image
# Then use RGB image to find object instances and their features
# Put the observations into the belief space (voxels) and return
def get_vox_preds(vlm, molmo_tools, robot_yaw, camera_pos, camera_pose, belief, obj_tp, obs, config, obstacle_map, camera_intrinsic_mat, feature=None, iteration=0, old_pix_coords=None):
    """
    Function to get predicted value of existence at each voxel (for an object type)
    given observation

    Inputs:
        vlm:                    VLM (prismatic) which can be used for object detection
        molmo_tools:            Contains molmo model and processor which is used for finding pixels cooresponding
                                    to object instances within the current observation
        robot_yaw:              Current yaw of the robot in the simulator map frame
        camera_pos:             Current position of the camera in the simulator map frame
        belief:                 Object belief for the type of object specified in 'obj_tp'
        obj_tp:                 The type of object to generate predictions about
        rgb_image:              The current RGB image observation
        depth_image:            The current DEPTH image observation
        config:                 The configuration data of the current task
        obstacle_map:           An instance of TSDFPlanner, where here the tsdf volume is used for predicting occupied
                                    regions in space
        camera_intrinsic_mat:   The camera intrinsic matrix used for coordinate transforms
                                    from the pixel frame, to the simulator map frame
        feature:                If not 'None' this string represents the feature of the object
                                    to generate predictions about
        iteration:              The current simulator step
        old_pix_coords:         For feature updating -- use pixel coords from main object detection

    Outputs:
    return voxel_preds, feature_vox_ret_vals, (len(pix_coords) > 0), pix_coords, real_world_coords
        voxel_preds:            A numpy array with the same dimensions as 'belief', which represents
                                    the predicted likelihood of the information being correct at each
                                    voxel given the current observation
        feature_vox_ret_vals:   A list of pairs (voxel coordinate, value) which represents the predicted
                                    value of 'feature' where instances were found (i.e. for each instance
                                    find the voxel coordinate, and feature value)
        (len(pix_coords)>0):    Equivalent to "found_obj". Used to determine if an instance was found
                                    in functions that call this function
        pix_coords:             A list of pixel coordinates, corresponding to found instances from the current
                                    observation
        real_world_coords:      The pix_coords but projected into the simulator map frame, based on the depth
                                    image values.
    """

    print("Robot Yaw: ", robot_yaw)

    voxel_preds = torch.ones(belief.p.shape).to(torch.device(config['bel_params']['torch_device']))
    voxel_preds *= -1

    # Make 0 in all visible voxels
    resolution = belief.map_params['res']
    dim = belief.map_params['dim']
    vol_origin = belief.map_params['vol_origin']

    print("RESOLUTION IN GET_VOX_PRED: ", resolution)
    print("AND DIM: ", dim)

    print("Starting FOV")
    voxel_preds = get_fov_from_depth_image(camera_pos, camera_pose, obs['depth'], voxel_preds, resolution, belief.map_params['z_res'], dim, vol_origin, config, camera_intrinsic_mat)

    # Get object detection
    pix_coords, real_world_coords, feature_ret_vals, logits = obj_detection_molmo(vlm, molmo_tools, camera_intrinsic_mat, obj_tp, obs, config, camera_pos, camera_pose, feature=feature, old_pix_coords=old_pix_coords)

    # Based on real world coordinates from object detection
    # Get voxel coordinates (in belief space) to return for updating
    # the belief
    print("First BOXES")
    feature_vox_ret_vals = []
    for i in range(len(pix_coords)):
        x, y, z = real_world_coords[i]
        print("REAL WORLD: ", x,y,z)
        score = logits[i]

        map_resolution = belief.map_params['res']
        map_dim = belief.map_params['dim']
        vol_origin = belief.map_params['vol_origin']
        z_resolution = belief.map_params['z_res']
        vxyz = world_to_map(np.array([x,y,z]), vol_origin, map_resolution, z_resolution, map_dim)

        if not np.isnan(z):

            print("MAP: ", vxyz)

            # Put score in prediction output
            if config['observation_calc_params']['use_model_score']:
                voxel_preds[vxyz[0], vxyz[1], vxyz[2]] = score
            else:
                voxel_preds[vxyz[0], vxyz[1], vxyz[2]] = config['observation_calc_params']['prob_correct_given_observed']

            if feature != None:
                feature_vox_ret_vals.append([vxyz, feature_ret_vals[i]])

    return voxel_preds, feature_vox_ret_vals, (len(pix_coords) > 0), pix_coords, real_world_coords
