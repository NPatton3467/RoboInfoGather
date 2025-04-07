import json
import numpy as np
import cv2
from RoboInfoGather.map_utils import *
from groundingdino.util.inference import predict
import groundingdino.datasets.transforms as T
from RoboInfoGather.MCTS_planner import Loc

from PIL import Image

import torch
import torchvision.transforms.functional as TF

from scipy.spatial.transform import Rotation as R
import skimage.measure

from matplotlib import pyplot as plt 
import glob

import openai
from openai import OpenAI
f = open('/robodata/user_data/npatt/explore-eqa/RoboInfoGather/openaikey.txt', 'r')
openai_api_key = f.read().rstrip('\n')
f.close()

import base64
from io import BytesIO

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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

def get_world_coords_from_depth(x, y, depth, camera_pos, camera_pose, camera_intrinsic_mat):
    cx = camera_intrinsic_mat[0][2]
    cy = camera_intrinsic_mat[1][2]
    fx = camera_intrinsic_mat[0][0]
    fy = camera_intrinsic_mat[1][1]
    
    camera_coords_z = depth
    camera_coords_x = (x-cx)*camera_coords_z/fx
    camera_coords_y = (y-cy)*camera_coords_z/fy

    debug = False
    if debug:
        print("CX: ", cx)
        print("CY: ", cy)
        print("FX: ", fx)
        print("FY: ", fy)
        print("X: ", x)
        print("Y: ", y)

    # Translate 3D coordinates to global frame
    c_coord = np.array([camera_coords_x, camera_coords_y, camera_coords_z, 1])

    # Coordinate transform
    c_coord2 = np.array([c_coord[2], -c_coord[0], -c_coord[1], 1])

    # Rotate to yaw 
    if debug:
        print("Camera Pose: ", camera_pose)
        print("Camera Position: ", camera_pos)
        print("Coordinates in Camera Frame: ", c_coord)
        print("Coordinates in Camera Frame2: ", c_coord2)

        print("Apply Camera Pose to 1: ", np.matmul(camera_pose, c_coord))
        print("Apply Camera Pose to 2: ", np.matmul(camera_pose, c_coord2))

    world_coords = np.matmul(camera_pose, c_coord2)
    return world_coords[0:3]

def get_fov_from_depth_image(camera_pos, camera_pose, raw_depth_image, voxel_preds, resolution, z_res, dim, vol_origin, config, cam_int_mat):
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
                if False:
                    print("Current Depth: ", cur_depth)
                    print("World Coordinates: ", world_coords)
                    print("Voxel Coordinates: ", v_xyz)

                if not np.isnan(world_coords[2]):
                    voxel_preds[v_xyz[0], v_xyz[1], v_xyz[2]] = config['observation_calc_params']['prob_occ_given_obs_free']

                cur_depth += (resolution / 2)


    print("Done Depth Loop")
    
    return voxel_preds
                

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


def obj_detection(vlm, cam_int_mat, dino_model, obj_tp, rgb_img, depth_img, config, camera_pos, camera_pose, feature=None):
    img = np.copy(rgb_img)
    #Image should be torch tensor
    img = Image.fromarray(img).convert('RGB')
    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    img, _ = transform(img, None)


    TEXT_PROMPT = f'{obj_tp}'
    BOX_THRESHOLD = 0.6
    TEXT_THRESHOLD = 0.25

    with torch.no_grad():
        boxes, logits, _ = predict(
            model=dino_model,
            image=img,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

    feature_ret_vals = [] 
    real_world_coords = []
    for box in boxes:
        cropped_img = np.copy(rgb_img)
        p_x = int(box[0] * cropped_img.shape[1])
        p_y = int(box[1] * cropped_img.shape[0])

        cur_depth = depth_img[p_y, p_x]

        real_world_coords.append(get_world_coords_from_depth(p_x, p_y, cur_depth, camera_pos, camera_pose, cam_int_mat))
        if feature != None:
            x_min = min(cropped_img.shape[1], max(0, int((box[0] - box[2]) * cropped_img.shape[1])))
            x_max = min(cropped_img.shape[1], max(0, int((box[0] + box[2]) * cropped_img.shape[1])))
            y_min = min(cropped_img.shape[0], max(0, int((box[1] - box[3]) * cropped_img.shape[0])))
            y_max = min(cropped_img.shape[0], max(0, int((box[1] + box[3]) * cropped_img.shape[0])))

            # Don't want to crop to practically 0 pixels
            print("Pre-Cropped Image Shape: ", cropped_img.shape)
            if (x_max - x_min) >= 5 and (y_max - y_min) >= 5:
                cropped_img = cropped_img[y_min:y_max, x_min:x_max, :]
           
            print("Post-Cropped Image Shape: ", cropped_img.shape)
            print("x_max: ", x_max)
            print("x_min: ", x_min)
            print("y_max: ", y_max)
            print("y_min: ", y_min)
            img = Image.fromarray(cropped_img).convert('RGB')
            print("RGB Image Shape: ", img.size)


            # Path to your image
            image_path = "./RoboInfoGather/feature_pre_prompt_figs/fridge_material.png"

            # Getting the Base64 string
            fridge_image = encode_image(image_path)

            message_1 = {
                        "role": "user",
                        "content": [
                                {
                                    "type": "text",
                                    "text": "Example 1\nObject Type: Fridge\nFeature to evaluate: material\n\nValue: Stainless Steel"
                                },
                                {
                                    "type": "image_url",
                                    "image_url":{"url" : f"data:image/png;base64,{fridge_image}"}
                                }
                            ]
                    }

            # Path to your image
            image_path = "./RoboInfoGather/feature_pre_prompt_figs/blanket_folded.png"

            # Getting the Base64 string
            blanket_image = encode_image(image_path)
            message_2 = {
                        "role": "user",
                        "content": [
                                {
                                    "type": "text",
                                    "text": "Example 2\nObject Type: Blanket\nFeature to evaluate: folded\n\nValue: Yes"
                                },
                                {
                                    "type": "image_url",
                                    "image_url":{"url" : f"data:image/png;base64,{blanket_image}"}
                                }
                            ]
                    }
            
            # Path to your image
            image_path = "./RoboInfoGather/feature_pre_prompt_figs/curtain_colour_white.png"

            # Getting the Base64 string
            curtain_image = encode_image(image_path)
            message_3 = {
                        "role": "user",
                        "content": [
                                {
                                    "type": "text",
                                    "text": "Example 3\nObject Type: Curtain\nFeature to evaluate: colour\n\nValue: White"
                                },
                                {
                                    "type": "image_url",
                                    "image_url":{"url" : f"data:image/png;base64,{curtain_image}"}
                                }
                            ]
                    }
            
            # Path to your image
            image_path = "./RoboInfoGather/feature_pre_prompt_figs/door_closed.png"

            # Getting the Base64 string
            door_image = encode_image(image_path)
            message_4 = {
                        "role": "user",
                        "content": [
                                {
                                    "type": "text",
                                    "text": "Example 4\nObject Type: Door\nFeature to evaluate: open\n\nValue: closed"
                                },
                                {
                                    "type": "image_url",
                                    "image_url":{"url" : f"data:image/png;base64,{door_image}"}
                                }
                            ]
                    }
            
            # Path to your image
            image_path = "./RoboInfoGather/feature_pre_prompt_figs/overhead_light_on.png"

            # Getting the Base64 string
            light_image = encode_image(image_path)
            message_5 = {
                        "role": "user",
                        "content": [
                                {
                                    "type": "text",
                                    "text": "Example 5\nObject Type: Light\nFeature to evaluate: turned on\n\nValue: on"
                                },
                                {
                                    "type": "image_url",
                                    "image_url":{"url" : f"data:image/png;base64,{light_image}"}
                                }
                            ]
                    }
            
            buffered = BytesIO()
            print("Image Size: ", img.size)
            img.save(buffered, format="JPEG")
            cur_img_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
            cur_message = {
                        "role": "user",
                        "content": [
                                {
                                    "type": "text",
                                    "text": f"Now please evaluate the following feature given the above examples, the current object type, and image.\nObject Type: {obj_tp}\nFeature to evaluate: {feature}\n\nPlease responde with only the value below\nValue: "
                                },
                                {
                                    "type": "image_url",
                                    "image_url":{"url" : f"data:image/png;base64,{cur_img_encoded}"}
                                }
                            ]
                    }
            messages = [
                message_1,
                message_2,
                message_3,
                message_4,
                message_5,
                cur_message
            ]
    
            # Query VLM
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
              model="gpt-4o-mini-2024-07-18",
              messages=messages,
              max_tokens=300,
            )
           
            response = response.choices[0].message.content

            print("Response: ", response)
            print("Cropped Image Shape: ", cropped_img.shape)

            feature_ret_vals.append(response)

    return boxes, real_world_coords, feature_ret_vals, logits

def get_vox_preds(vlm, robot_yaw, camera_pos, camera_pose, belief, obj_tp, rgb_image, depth_image, dino_model, config, obstacle_map, camera_intrinsic_mat, feature=None, iteration=0):
    """
    Function to get predicted value of existence at each voxel (for an object type) 
    give observation

    :param:


    :returns: np.array with same size as belief, where voxels within observation are updated based on 
    predicted value of object existence.
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
    voxel_preds = get_fov_from_depth_image(camera_pos, camera_pose, depth_image, voxel_preds, resolution, belief.map_params['z_res'], dim, vol_origin, config, camera_intrinsic_mat)
   
    # Get object detection
    boxes, real_world_coords, feature_ret_vals, logits = obj_detection(vlm, camera_intrinsic_mat, dino_model, obj_tp, rgb_image, depth_image, config, camera_pos, camera_pose, feature=feature)
    
    print("First BOXES")
    feature_vox_ret_vals = []
    for i in range(len(real_world_coords)):
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
            
    return voxel_preds, feature_vox_ret_vals
