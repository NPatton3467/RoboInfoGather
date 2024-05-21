import numpy as np
import cv2
from RoboInfoGather.map_utils import *
from groundingdino.util.inference import predict
import groundingdino.datasets.transforms as T
from RoboInfoGather.MCTS_planner import Loc

from ram.models import ram
from ram import inference_ram, inference_tag2text, inference_ram_openset
from ram import get_transform

from PIL import Image

import torch
import torchvision.transforms.functional as TF

from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt 

# Setup global ram model
ram_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ram_checkpoint = '/robodata/user_data/npatt/OmniGibson/RoboInfoGather/pretrained/ram_plus_swin_large_14m.pth'
ram_img_size = 384
ram_model = ram(pretrained=ram_checkpoint, vit='large', image_size=ram_img_size)
ram_model.eval()
ram_model = ram_model.to(ram_device)

def get_ir_o_map(obstacle_map, new_size, dilation_radius_pre=12, erosion_radius_post=2):
    ir_o_map = np.where(obstacle_map > 0, 100, 0)
    ir_o_map = ir_o_map.astype('uint8')
    ir_o_map = cv2.dilate(ir_o_map, np.ones((dilation_radius_pre,dilation_radius_pre)))

    ir_o_map = cv2.resize(ir_o_map, (new_size, new_size))
    ir_o_map = np.where(ir_o_map > 0, 100, 0).astype('uint8')

    #ir_o_map = cv2.erode(ir_o_map, np.ones((erosion_radius_post,erosion_radius_post)))

    return ir_o_map


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


def get_fov(current_location, config, camera_params, obstacle_map, belief, debug_print=True):
    min_angle = camera_params['min_angle']
    max_angle = camera_params['max_angle']
    min_v_dist = camera_params['min_visual_distance']
    max_v_dist = camera_params['max_visual_distance']

    angle_delta = config['rf_params']['angle_delta']
    dist_delta = belief.map_params['res'] * 0.8

    angle = min_angle

    # Use inflated and resized obstacle map to not see through walls?
    inflated_resized_obstacle_map = get_ir_o_map(obstacle_map.obstacles, belief.map_params['size'])

    fov = []
    while angle < max_angle:
        dist = min_v_dist
        if debug_print:
            print(len(fov))
        while dist < max_v_dist:
            # Get node that corresponds to angle and dist (relative to robot)
            x, y = get_new_node(current_location, dist, angle, belief)

            # Skip if already added
            if (x, y) in fov:
                dist += dist_delta
                continue

            # Check that x,y are within map bounds
            x_max = belief.map_params['size']
            y_max = belief.map_params['size']
            b_xy = world_to_map(np.array([x,y]), belief.map_params['res'], belief.map_params['size'])
            if b_xy[0] not in range(0, x_max) or b_xy[1] not in range(0, y_max):
                break

            # Can't see through obstacles so break
            if inflated_resized_obstacle_map[b_xy[0], b_xy[1]]:
                break

            fov.append((x,y))

            # Increment distance
            dist += dist_delta

        # Increment angle
        angle += angle_delta

    return fov

def quat_to_rot(quat):
    return R.from_quat(quat).as_matrix()


def get_real_coords(x, y, camera_pos, camera_ori, depth_image, camera_intrinsic_mat):
    """
    This function uses the depth camera pixel values, camera intrinsic matrix, and camera position
    to translate pixel values to real world cooridnates
    """

    # Calculate 3D coordinates in camera frame
    cx = camera_intrinsic_mat[0][2]
    cy = camera_intrinsic_mat[1][2]
    fx = camera_intrinsic_mat[0][0]
    fy = camera_intrinsic_mat[1][1]

    print(x, y)
    x_idx = int(x * depth_image.shape[0])
    y_idx = int(y * depth_image.shape[1])

    print(x_idx, y_idx)

    camera_coords_z = depth_image[x_idx,y_idx]
    print(camera_coords_z)

    camera_coords_x = (x-cx)*camera_coords_z/fx
    camera_coords_y = (y-cy)*camera_coords_z/fy

    # Translate 3D coordinates to global frame
    c_coord = np.array([camera_coords_x, camera_coords_y, camera_coords_z])

    print(c_coord)

    # Set up rotation Matrix based off of camera location
    Rotation = quat_to_rot(camera_ori)

    # Set up translation vector based off of actual camera position 
    translation = camera_pos

    # Adjust yaw 90 deg for camera
    theta = np.deg2rad(90)
    Adj_Rot = np.array([
        [np.cos(theta), -1*np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0,0,1]
    ])

    world_coords = np.matmul(Adj_Rot, np.matmul(Rotation, c_coord)) + translation
    
    print(world_coords)

    return world_coords


def obj_detection(dino_model, obj_tp, state, feature):
    img = np.array(state['robot0:eyes:Camera:0']['rgb'])
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

    if feature is None:
        TEXT_PROMPT = f'{obj_tp}'
    else:
        TEXT_PROMPT = f'{obj_tp} with {feature}'

    BOX_THRESHOLD = 0.5
    TEXT_THRESHOLD = 0.25

    with torch.no_grad():
        boxes, logits, _ = predict(
            model=dino_model,
            image=img,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

    return boxes, logits


def get_new_loc(current_loc, dist, angle, res, size):
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


def get_all_object_detections(state, dino_model):
    img = np.array(state['robot0:eyes:Camera:0']['rgb'])
    #Image should be torch tensor
    img = Image.fromarray(img).convert('RGB')
    transform = get_transform(ram_img_size)
    img = transform(img)
    img = img.unsqueeze(0).to(ram_device)
       
    with torch.no_grad():
        TEXT_PROMPT = inference_ram_openset(img, ram_model)

    print("START", TEXT_PROMPT, "END")
    if TEXT_PROMPT == "" or TEXT_PROMPT == " ":
        return [], []

    TEXT_PROMPT_LIST = TEXT_PROMPT.split('|')
    TEXT_PROMPT = ''
    for i in range(len(TEXT_PROMPT_LIST)):
        if i == len(TEXT_PROMPT_LIST) - 1:
            TEXT_PROMPT += TEXT_PROMPT_LIST[i].strip(' ')
        else:
            TEXT_PROMPT += TEXT_PROMPT_LIST[i].strip(' ') + ", "

    print(TEXT_PROMPT)


    # Need to reform image for DINO
    img = np.array(state['robot0:eyes:Camera:0']['rgb'])
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

    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    with torch.no_grad():
        boxes, logits, object_names = predict(
            model=dino_model,
            image=img,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

    return boxes, object_names


def get_vlm_prediction(state, obj_tp, obj_tp2):
    f = open('VLM_occlusion_prompt.txt', 'r')
    pre_prompt = f.read()
    f.close()

    # Append prompt with current example
    prompt = pre_prompt + f"Now, with the provided image, is it likely that an object of type [object_type_1 = {obj_tp}] is occluded by and object of [object_type_2 = {obj_tp2}]?"

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                f"{prompt}",
                *map({"image": state['robot0:eyes:Camera:0']['rgb'], "resize": 128}, base64Frames[1000:1500:20]),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        #"max_tokens": 200,
    }

    result = client.chat.completions.create(**params)
    response = result.choices[0].message.content

    # Parse response and assign bool
    bool_response = True if response == "True" else False

    return bool_response



def predict_unlikely_occluded_voxels(camera_pos, camera_rpy, obj_tp, state, config, obstacle_map, belief, dino_model, feature=None):
    """
    Function to get a set of voxels corresponding to occlusions in current image that are unlikely to contain an
    instance of the desired object type

    Steps:
        1. Scan robot fov (x, y only) and pick voxels that are occluded based on obstacle map
            a. Save into regions based off of angle (i.e. adjacent angles will occlusion will correspond to same region)
        2. Based on these voxels determine which object is blocking in image
            a. Do based on regions of voxels. Average x,y location of region is sample, then project to image space and
            detect object there
        3. Query VLM with image + "is it unlikely that obj_tp is occluded behind obj_tp2 in this image"
        4. If "yes" to above save voxel set for that object type
    """

    # Find occluded voxels
    camera_params = config['camera_params']
    rf_params = config['rf_params']
    min_angle = camera_params['min_angle']
    max_angle = camera_params['max_angle']
    min_v_dist = camera_params['min_visual_distance']
    max_v_dist = camera_params['max_visual_distance']

    angle_delta = rf_params['angle_delta']
    dist_delta = rf_params['dist_delta']

    angle = min_angle
    dist = min_v_dist

    loc = Loc(camera_pos[0], camera_pos[1], camera_rpy[2])

    occluded_voxels = {}
    appending_to_group = False
    appending_to_group_angle = -1
    appending_to_group_count = 0 # This is for "erosion" -> remove first and last angle in group
    print("Starting Angle Sweep")
    while angle < max_angle:
        found_occlusion_in_current_angle = False
        while dist < max_v_dist:
            # Get node that corresponds to angle and dist (relative to robot)
            x, y = get_new_loc(loc, dist, angle, belief.map_params['res'], belief.map_params['size'])
            o_map_xy = world_to_map(np.array([x, y]), obstacle_map.resolution, obstacle_map.size)

            # Check that x,y are within map bounds
            x_max = belief.map_params['size']
            y_max = belief.map_params['size']
            if o_map_xy[0] not in range(0, x_max) or o_map_xy[1] not in range(0, y_max):
                break

            # Can't see through obstacles so break
            if obstacle_map.obstacles[o_map_xy[0], o_map_xy[1]] > 0:
                found_occlusion_in_current_angle = True
                # Check if current angle is in occluded voxels
                if angle in occluded_voxels:
                    if appending_to_group_count in occluded_voxels[angle]:
                        if (x,y) not in occluded_voxels[angle][appending_to_group_count]:
                            occluded_voxels[angle][appending_to_group_count].append((x,y))
                    else:
                        occluded_voxels[angle][appending_to_group_count] = [(x,y)]

                # Check if currently in a group
                elif appending_to_group:
                    if appending_to_group_count in occluded_voxels[appending_to_group_angle]:
                        if (x,y) not in occluded_voxels[appending_to_group_angle][appending_to_group_count]:
                            occluded_voxels[appending_to_group_angle][appending_to_group_count].append((x,y))
                    else:
                        occluded_voxels[appending_to_group_angle][appending_to_group_count] = [(x,y)]

                # Else need to start new group
                else:
                    appending_to_group = True
                    appending_to_group_angle = angle

                    occluded_voxels[appending_to_group_angle] = {appending_to_group_count : [(x,y)]}


            # Increment distance
            dist += dist_delta

        # Increment angle
        angle += angle_delta
        appending_to_group_count += 1

        if not found_occlusion_in_current_angle:
            appending_to_group = False
            appending_to_group_angle = -1
            appending_to_group_count = 0


    print("Starting Angle Erosion")
    # Now erode voxels at angle boundary of groups (first and last)
    # The idea here is we want to make sure that an object whose center is *anywhere* in that voxel
    # is occluded, and we consider *only* those such voxels
    non_eroded_voxels = occluded_voxels
    occluded_voxels = {}
    for group in non_eroded_voxels:
        if len(non_eroded_voxels[group]) < 3:
            continue

        else:
            temp_dict = non_eroded_voxels[group]
            _ = temp_dict.pop(len(temp_dict) - 1)
            _ = temp_dict.pop(0)

            occluded_voxels[group] = []
            for sub_group in temp_dict:
                occluded_voxels[group] += temp_dict[sub_group]

    
    # Occluded voxels found
    # Now need to reason about probability of existence behind occlusions

    # First get all object predictions
    boxes, obj_names = get_all_object_detections(state, dino_model)


    print("Starting Grouping")
    # For each group in the occluded voxels, project into image space and find object that is causing occlusion
    return_voxels = []
    for group in occluded_voxels:
        x_avg = 0
        y_avg = 0

        # Calculate average xy postion
        for x, y in occluded_voxels[group]:
            if temp_map[x,y] == 1:
                x_avg += x / len(occluded_voxels[group])
                y_avg += y / len(occluded_voxels[group])

        if x_avg != 0.0 and y_avg != 0.0:
            # Project into pixel space
            rvec = camera_angle_mat
            tvec = camera_pos
            cameraMat = camera_intrinsic_mat
            img_point = cv2.projectPoints([x_avg, y_avg], rvec, tvec, cameraMat)

            # Check if point within bbox
            correct_bbox = None
            correct_name = ""
            for i in range(boxes):
                bbox = boxes[i]
                name = obj_names[1]
                # Grounding Dino format is cxcywh
                assert False # Check that this is the same for RAM GROUNDING DINO
                
                # Check if within bbox
                if img[0] < bbox[0] + (bbox[2]/2) and img[0] > bbox[0] - (bbox[2]/2) and \
                    img[1] < bbox[1] + (bbox[3]/2) and img[1] > bbox[1] - (bbox[3]/2):

                    correct_bbox = bbox
                    correct_name = name

                    break

            # If found a corresponding bounding box
            # Calculate likelihood of existence behind occlusion with vlm
            if correct_bbox is not None:
                assert False # TODO 

                # ASK VLM: Given img, is it likely that obj_tp is behind [correct_name]?
                
                likely = get_vlm_prediction(state, obj_tp, correct_name)

                if not likely:
                    if return_voxels == []:
                        return_voxels = occluded_voxels[group]
                    else:
                        return_voxels += occluded_voxels[group]


    return return_voxels



def get_vox_preds(camera_pos, camera_ori, camera_rpy, belief, obj_tp, state, dino_model, config, obstacle_map, camera_intrinsic_mat, feature=None):
    """
    Function to get predicted value of existence at each voxel (for an object type) 
    give observation

    :param:


    :returns: np.array with same size as belief, where voxels within observation are updated based on 
    predicted value of object existence.
    """

    voxel_preds = np.ones(belief.p.shape)
    voxel_preds *= -1
    
    # Make 0 in all visible voxels
    camera_angle_mat = quat_to_rot(camera_ori)
    loc = Loc(camera_pos[0], camera_pos[1], camera_rpy[2])# - np.deg2rad(90))
    print("Camera Angle", np.arccos(camera_angle_mat[0][0]) - np.deg2rad(90), camera_rpy[2])
    fov = get_fov(loc, config, config['camera_params'], obstacle_map, belief)
    for x, y in fov:
        vxy = world_to_map(np.array([x, y]), map_resolution=belief.map_params['res'], map_size=belief.map_params['size'])
        for z in range(belief.z_dim):
            # Set 0 for whole z_dim
            voxel_preds[vxy[0], vxy[1], z] = 0


    # Get the set object bounding boxes and confidence scores for object types/features from state
    boxes, logits = obj_detection(dino_model, obj_tp, state, feature)

    # Get the corresponding voxels
    for i in range(len(boxes)):
        bbox = boxes[i]
        score = 1/(1+np.exp(-1*logits[i]))
        # Get bounding box center 
        # Grounding Dino format is cxcywh
        cx = bbox[0]
        cy = bbox[1]

        # Get xyz coordinates from image and depth
        x, y, z = get_real_coords(cx, cy, camera_pos, camera_ori, state['robot0:eyes:Camera:0']['depth_linear'], camera_intrinsic_mat)

        # Translate to map coords and add to prediction
        xy = [x, y]
        map_resolution = belief.map_params['res']
        map_size = belief.map_params['size']
        vxy = world_to_map(xy, map_resolution, map_size)

        vx = vxy[0]
        vy = vxy[1]
        vz = int(z / map_resolution)

        # Put score in prediction output
        voxel_preds[vx, vy, vz] = score


    # Predict score for occluded regions
    low_likelihood_voxels = predict_unlikely_occluded_voxels(camera_pos, camera_rpy, obj_tp, state, config, obstacle_map, belief, dino_model, feature)

    for vox in low_likelihood_voxels:
        # Loop throught z-dim
        for z in range(belief.z_dim):
            # Make sure we're not contradiction previous observation scores
            vxy = world_to_map(np.array([vox[0], vox[1]], map_resolution=belief.map_params['res'], map_size=belief.map_params['size']))
            if voxel_preds[vxy[0], vxy[1], z] == 0:
                voxel_preds[vxy[0], vxy[1], z] = config['observation_calc_params']['dne_occluded_prob']

    print("Done get_vox_preds")

    return voxel_preds
