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

def get_world_coords_from_depth(x, y, depth, camera_pos, robot_yaw, camera_intrinsic_mat):
    cx = camera_intrinsic_mat[0][2]
    cy = camera_intrinsic_mat[1][2]
    fx = camera_intrinsic_mat[0][0]
    fy = camera_intrinsic_mat[1][1]
    
    camera_coords_z = depth
    camera_coords_x = (x-cx)*camera_coords_z/fx
    camera_coords_y = (y-cy)*camera_coords_z/fy

    # Translate 3D coordinates to global frame
    c_coord = np.array([camera_coords_x, camera_coords_y, camera_coords_z])

    # Set up translation vector based off of actual camera position 
    translation = camera_pos

    # Coordinate transform
    c_coord = np.array([c_coord[2], -c_coord[0], -c_coord[1]])

    # Rotate to yaw 
    Rotation = np.array([
        [np.cos(robot_yaw), -np.sin(robot_yaw), 0],
        [np.sin(robot_yaw), np.cos(robot_yaw), 0],
        [0,0,1]])
    world_coords = np.matmul(Rotation, c_coord)

    # Translate
    world_coords += translation

    return world_coords

def get_fov_from_depth_image(camera_pos, robot_yaw, raw_depth_image, voxel_preds, resolution, z_res, size, config, cam_int_mat):
    # Max pool to decrease image size
    depth_image = skimage.measure.block_reduce(raw_depth_image, (16,16), np.min)
    print("Got Depth Image... Shape: ", depth_image.shape)

    # Set to zero up to obstacle
    for p_x in range(depth_image.shape[1]):
        for p_y in range(depth_image.shape[0]):
            #print(p_x, p_y)
            max_depth = depth_image[p_y, p_x]
            if max_depth >= size * resolution:
                continue

            cur_depth = 0
            while cur_depth < max_depth:
                world_coords = get_world_coords_from_depth(p_x, p_y, cur_depth, camera_pos, robot_yaw, cam_int_mat)

                # Set voxel pred location to 0 here
                v_xy = world_to_map(np.array([world_coords[0],world_coords[1]]), resolution, size)

                if not np.isnan(world_coords[2]):
                    vz = int(world_coords[2] / z_res)

                    if v_xy[0] in range(0, size) and v_xy[1] in range(0, size) and vz in range(0, voxel_preds.shape[2]):
                        voxel_preds[v_xy[0], v_xy[1], vz] = config['observation_calc_params']['prob_occ_given_obs_free']

                cur_depth += (resolution / 2)


    print("Done first Depth Loop")
    
    # Make sure obstacles are still set to -1
    for p_x in range(depth_image.shape[1]):
        for p_y in range(depth_image.shape[0]):
            cur_depth = depth_image[p_y, p_x]
            max_depth = cur_depth + 2
            while cur_depth < max_depth:
                world_coords = get_world_coords_from_depth(p_x, p_y, cur_depth, camera_pos, robot_yaw, cam_int_mat)

                # Set voxel pred location to 0 here
                v_xy = world_to_map(np.array([world_coords[0],world_coords[1]]), resolution, size)
                
                if not np.isnan(world_coords[2]):
                    vz = int(world_coords[2] / z_res)
                    
                    if v_xy[0] in range(0, size) and v_xy[1] in range(0, size) and vz in range(0, voxel_preds.shape[2]):
                        voxel_preds[v_xy[0], v_xy[1], vz] = -1

                cur_depth += (resolution / 2)
               

    print("Done second Depth Loop")
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


def obj_detection(vlm, cam_int_mat, dino_model, obj_tp, rgb_img, depth_img, config, camera_pos, robot_yaw, feature=None):
    img = rgb_img
    #Image should be torch tensor
    img = Image.fromarray(img).convert('RGB')
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
        cropped_img = rgb_img
        p_x = int(box[0] * cropped_img.shape[1])
        p_y = int(box[1] * cropped_img.shape[0])

        cur_depth = depth_img[p_y, p_x]

        real_world_coords.append(get_world_coords_from_depth(p_x, p_y, cur_depth, camera_pos, robot_yaw, cam_int_mat))
        if feature != None:
            x_min = int((box[0] - box[2]) * cropped_img.shape[1])
            x_max = int((box[0] + box[2]) * cropped_img.shape[1])
            y_min = int((box[1] - box[3]) * cropped_img.shape[0])
            y_max = int((box[1] + box[3]) * cropped_img.shape[0])

            # Don't want to crop to practically 0 pixels
            if (x_max - x_min) >= 5 and (y_max - y_min) >= 5:
                cropped_img = cropped_img[y_min:y_max, x_min:x_max, :]
            
            img = Image.fromarray(cropped_img).convert('RGB')
    
            # Query VLM
            prompt = f"Given the image and object type `{obj_tp}`, what is the value of the feature `{feature}`? Please respond with only the answer to the above question."

            response = vlm.generate(prompt, img)

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
    size = belief.map_params['size']

    print("RESOLUTION IN GET_VOX_PRED: ", resolution)
    print("AND SIZE: ", size)

    z_dim_max = belief.z_dim
    print("Starting FOV")
    voxel_preds = get_fov_from_depth_image(camera_pos, robot_yaw, depth_image, voxel_preds, resolution, belief.map_params['z_res'], size, config, camera_intrinsic_mat)
   
    # Get object detection
    boxes, real_world_coords, feature_ret_vals, logits = obj_detection(vlm, camera_intrinsic_mat, dino_model, obj_tp, rgb_image, depth_image, config, camera_pos, robot_yaw, feature=feature)
    
    print("First BOXES")
    feature_vox_ret_vals = []
    for i in range(len(real_world_coords)):
        x, y, z = real_world_coords[i]
        print("REAL WORLD: ", x,y,z)
        score = logits[i] 
        
        xy = [x, y]
        map_resolution = belief.map_params['res']
        map_size = belief.map_params['size']
        vxy = world_to_map(xy, map_resolution, map_size)

        vx = vxy[0]
        vy = vxy[1]

        if not np.isnan(z):
            vz = int(z / belief.map_params['z_res'])

            print("MAP: ", vx, vy, vz)

            # Put score in prediction output
            if vx < voxel_preds.shape[0] and vx >= 0 and\
                    vy < voxel_preds.shape[1] and vy >= 0 and\
                    vz < voxel_preds.shape[2] and vz >= 0:
                if config['observation_calc_params']['use_model_score']:
                    voxel_preds[vx, vy, vz] = score
                else:
                    voxel_preds[vx, vy, vz] = config['observation_calc_params']['prob_correct_given_observed'] 

                if feature != None:
                    feature_vox_ret_vals.append([np.array([vx, vy, vz]), feature_ret_vals[i]])
            
    return voxel_preds, feature_vox_ret_vals
