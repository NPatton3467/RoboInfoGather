import json
import numpy as np
import cv2
from RoboInfoGather.map_utils import *
from groundingdino.util.inference import predict
import groundingdino.datasets.transforms as T
from RoboInfoGather.MCTS_planner import Loc

#from omnigibson.utils.vision_utils import *

#from ram.models import ram
#from ram import inference_ram, inference_tag2text, inference_ram_openset
#from ram import get_transform

from PIL import Image

import torch
import torchvision.transforms.functional as TF
#from segment_anything import sam_model_registry, SamPredictor

from scipy.spatial.transform import Rotation as R
import skimage.measure

from matplotlib import pyplot as plt 
import glob

# Setup global ram model
#ram_device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
#ram_checkpoint = '/robodata/user_data/npatt/OmniGibson/RoboInfoGather/pretrained/ram_plus_swin_large_14m.pth'
#ram_img_size = 384
#ram_model = ram(pretrained=ram_checkpoint, vit='large', image_size=ram_img_size)
#ram_model.eval()
#ram_model = ram_model.to(ram_device)

#SAM_ENCODER_VERSION = "vit_h"
#DEVICE = torch.device('cuda:5')
#SAM_CHECKPOINT_PATH = '/robodata/user_data/npatt/OmniGibson/SAM/weights/sam_vit_h_4b8939.pth'
#sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
#sam_predictor = SamPredictor(sam)

def get_min_depth_sam(bbox, depth_image, mask, left):
    if left: # Find left most pixel in mask
        sy = int(bbox[1] * depth_image.shape[0])
        x = -1
        y = -1
        while x == -1 and y == -1:
            for sx in range(depth_image.shape[1]):
                if sy >= depth_image.shape[0] - 1:
                    x = sx + 3
                    y = depth_image.shape[0] - 1
                    break
                if mask[sy,sx] > 0:
                    x = sx + 3
                    y = sy
                    break
            sy += 1

        x = min(depth_image.shape[1] - 1, x)

        return x/depth_image.shape[1], y/depth_image.shape[0], depth_image[y,x]

    else: # Find left most pixel in mask
        sy = int(bbox[1] * depth_image.shape[0])
        x = -1
        y = -1
        while x == -1 and y == -1:
            for sx in range(depth_image.shape[1]):
                if sy >= depth_image.shape[0] - 1:
                    x = sx + 3
                    y = depth_image.shape[0] - 1
                    break
                if mask[sy,depth_image.shape[1] - sx - 1] > 0:
                    x = depth_image.shape[1] - sx - 1 - 3
                    y = sy
                    break
            sy += 1

        x = max(0, x)
        return x/depth_image.shape[1], y/depth_image.shape[0], depth_image[y,x]

def get_real_coords_sam(bbox, robot_yaw, camera_pos, depth_image, mask, camera_intrinsic_mat, resolution, left=True):
    """
    This function uses the depth camera pixel values, camera intrinsic matrix, and camera position
    to translate pixel values to real world cooridnates
    """
    # Calculate 3D coordinates in camera frame
    cx = camera_intrinsic_mat[0][2]
    cy = camera_intrinsic_mat[1][2]
    fx = camera_intrinsic_mat[0][0]
    fy = camera_intrinsic_mat[1][1]

    x, y, depth = get_min_depth_sam(bbox, depth_image, mask, left=left)
    camera_coords_z = depth

    camera_coords_x = (x-cx)*camera_coords_z/fx
    camera_coords_y = (y-cy)*camera_coords_z/fy

    # Translate 3D coordinates to global frame
    c_coord = np.array([camera_coords_x, camera_coords_y, camera_coords_z])

    # Set up translation vector based off of actual camera position
    translation = camera_pos
    c_coord = np.array([c_coord[2], -c_coord[0], -c_coord[1]])

    #world_coords = np.matmul(Adj_Rot, np.matmul(Rotation, c_coord)) + translation
    Rotation = np.array([
        [np.cos(robot_yaw), -np.sin(robot_yaw), 0],
        [np.sin(robot_yaw), np.cos(robot_yaw), 0],
        [0,0,1]])
    world_coords = np.matmul(Rotation, c_coord)

    world_coords += translation

    return world_coords[0], world_coords[1], world_coords[2]

def get_centroid_coords_sam(bbox, robot_yaw, camera_pos, image, depth_image, camera_intrinsic_mat, resolution):
    # Check SAM output
    img = image[:,:,:3]
    sam_predictor.set_image(img)
    sbbox = np.array([
        (bbox[0] - bbox[2]/2) * image.shape[1],
        (bbox[1] - bbox[3]/2) * image.shape[0],
        (bbox[0] + bbox[2]/2) * image.shape[1],
        (bbox[1] + bbox[3]/2) * image.shape[0]])
    mask, score, logit = sam_predictor.predict(box = sbbox, multimask_output=False)

    mask = np.reshape(mask, (128,128,1))
    mask = np.where(mask == True, 1, 0)

    # Get the world_coords of the left of the bounding box
    lx, ly, lz = get_real_coords_sam(bbox, robot_yaw, camera_pos, depth_image, mask, camera_intrinsic_mat, resolution, left=True)

    # Get the world_coords of the right of the bounding box
    rx, ry, rz = get_real_coords_sam(bbox, robot_yaw, camera_pos, depth_image, mask, camera_intrinsic_mat, resolution, left=False)

    # Estimate centroid -- assume (lx,ly) as reference point, (rx, ry) lies on positive x-axis of reference
    dist = resolution / 2
    ang = np.arctan2(ry-ly, rx-lx)

    print("Angle: ", ang)

    n_point = np.array([dist, dist])

    #Rotate and translate
    rot = np.array([[np.cos(ang), -np.sin(ang)],
                    [np.sin(ang), np.cos(ang)]])
    n_point = np.matmul(rot, n_point) + np.array([lx,ly])

    return n_point[0], n_point[1], (lz+rz)/2.0, True


def get_ir_o_map(obstacle_map, new_size, dilation_radius_pre=13, erosion_radius_post=2):
    # ir_o_map = np.where(obstacle_map > 0, 100, 0)
    # ir_o_map = ir_o_map.astype('uint8')
    # ir_o_map = cv2.dilate(ir_o_map, np.ones((dilation_radius_pre,dilation_radius_pre)))

    # ir_o_map = cv2.resize(ir_o_map, (new_size, new_size))
    # ir_o_map = np.where(ir_o_map > 0, 100, 0).astype('uint8')

    # ir_o_map = cv2.erode(ir_o_map, np.ones((erosion_radius_post,erosion_radius_post)))

    # return ir_o_map

    local_obs_map = cv2.dilate(obstacle_map.obstacles, np.ones((dilation_radius_pre,dilation_radius_pre)))

    return local_obs_map


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

                    print(f"Voxel: ({v_xy[0]}, {v_xy[1]}, {vz})")
                    print(f"Vox Pred Shape: ", voxel_preds.shape)
                    
                    if v_xy[0] in range(0, size) and v_xy[1] in range(0, size) and vz in range(0, voxel_preds.shape[2]):
                        voxel_preds[v_xy[0], v_xy[1], vz] = config['observation_calc_params']['prob_occ_given_obs_free']

                cur_depth += (resolution / 2)


    # Make sure obstacles are still set to -1
    for p_x in range(depth_image.shape[1]):
        for p_y in range(depth_image.shape[0]):
            cur_depth = depth_image[p_y, p_x]
            max_depth = size * resolution
            while cur_depth < max_depth:
                world_coords = get_world_coords_from_depth(p_x, p_y, cur_depth, camera_pos, robot_yaw, cam_int_mat)

                # Set voxel pred location to 0 here
                v_xy = world_to_map(np.array([world_coords[0],world_coords[1]]), resolution, size)
                
                if not np.isnan(world_coords[2]):
                    vz = int(world_coords[2] / z_res)
                    
                    if v_xy[0] in range(0, size) and v_xy[1] in range(0, size) and vz in range(0, voxel_preds.shape[2]):
                        voxel_preds[v_xy[0], v_xy[1], vz] = -1

                cur_depth += (resolution / 2)
               

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
    #inflated_resized_obstacle_map = get_ir_o_map(obstacle_map, belief.map_params['size'])
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

def quat_to_rot(quat):
    return R.from_quat(quat).as_matrix()

def quat_from_rpy(rpy):
    return R.from_euler('xyz', rpy).as_matrix()

def get_min_depth(bbox, depth_image):
    x_min = max(0, int((bbox[0] - bbox[2]/2) * depth_image.shape[1]))
    x_max = min(int((bbox[0] + bbox[2]/2) * depth_image.shape[1]), depth_image.shape[1] - 1)
    y_min = max(0, int((bbox[1] - bbox[3]/2) * depth_image.shape[0]))
    y_max = min(int((bbox[1] + bbox[3]/2) * depth_image.shape[0]), depth_image.shape[0] - 1)

    print(f"GETMINDEPTH: {y_min}, {y_max}, {x_min}, {x_max}")
    return np.min(depth_image[y_min:y_max, x_min:x_max])

def get_real_coords(bbox, robot_yaw, camera_pos, depth_image, depth_linear_image, camera_intrinsic_mat, resolution):
    """
    This function uses the depth camera pixel values, camera intrinsic matrix, and camera position
    to translate pixel values to real world cooridnates
    """
    # Grounding Dino format is cxcywh
    x = bbox[0]
    y = bbox[1]

    # Calculate 3D coordinates in camera frame
    cx = camera_intrinsic_mat[0][2]
    cy = camera_intrinsic_mat[1][2]
    fx = camera_intrinsic_mat[0][0]
    fy = camera_intrinsic_mat[1][1]

    depth = get_min_depth(bbox, depth_image)
    depth_linear = get_min_depth(bbox, depth_linear_image)
    camera_coords_z = depth_linear
    print(camera_coords_z)

    camera_coords_x = (x-cx)*camera_coords_z/fx
    camera_coords_y = (y-cy)*camera_coords_z/fy

    # Translate 3D coordinates to global frame
    c_coord = np.array([camera_coords_x, camera_coords_y, camera_coords_z])

    print("Camera Coords")
    print(c_coord)

    # Set up rotation Matrix based off of camera location
    #Rotation = R.from_euler('xyz', camera_rpy).as_matrix()
    #Rotation = R.from_quat(camera_ori).as_matrix()
    #print("Rotation Matrix")
    #print(Rotation)

    # Set up translation vector based off of actual camera position 
    translation = camera_pos
    #print("Camera Position")
    #print(translation)

    # Reorient camera pixel coords with robot
    # Roll/Pitch/Yaw Angles in pixel coord frame
    # Robot = Pitch(90) * Yaw(90) * Pixel
    
    # Roll -90
    #Roll = np.array([[1, 0, 0],\
    #                [0, np.cos(np.deg2rad(-90)), -np.sin(np.deg2rad(-90))],\
    #                [0, np.sin(np.deg2rad(-90)), np.cos(np.deg2rad(-90))]])

    # Pitch 90
    Pitch = np.array([[np.cos(np.deg2rad(90)), 0, np.sin(np.deg2rad(90))],\
            [0,1,0],\
            [-np.sin(np.deg2rad(90)), 0, np.cos(np.deg2rad(90))]])

    # Yaw 90
    Yaw = np.array([[np.cos(np.deg2rad(-90)), -np.sin(np.deg2rad(-90)), 0],\
            [np.sin(np.deg2rad(-90)), np.cos(np.deg2rad(-90)), 0],\
            [0,0,1]])
    
    #c_coord = np.matmul(np.matmul(Pitch, Yaw), c_coord) 
    c_coord = np.array([c_coord[2], -c_coord[0], -c_coord[1]])

    #world_coords = np.matmul(Adj_Rot, np.matmul(Rotation, c_coord)) + translation
    Rotation = np.array([
        [np.cos(robot_yaw), -np.sin(robot_yaw), 0],
        [np.sin(robot_yaw), np.cos(robot_yaw), 0],
        [0,0,1]])
    world_coords = np.matmul(Rotation, c_coord)
    #print("Rotated camera Coords")
    #print(world_coords)

    #adj_rot = np.array([[-1,0,0], [0,-1,0],[0,0,1]])
    #world_coords = np.matmul(adj_rot, world_coords)
    #print("Rotation Adjusted Camera Coords")
    #print(world_coords)

    world_coords += translation
    
    #print("World Coords")
    #print(world_coords)


    # Try 2
    #uv1 = np.array([x, y, 1])
    #ray = np.matmul(np.linalg.inv(camera_intrinsic_mat), uv1)
    #print("Ray: ", ray)
    #ray = (ray / np.linalg.norm(ray)) * camera_coords_z
    #print("Length Corrected Ray: ", ray)
    #xyz_c = ray - translation
    #world_coords = np.matmul(np.linalg.inv(Rotation), xyz_c)
    
    return world_coords[0], world_coords[1], world_coords[2], depth, depth_linear

def get_centroid_coords(bbox, robot_yaw, camera_pos, camera_ori, depth_image, depth_linear_image, camera_intrinsic_mat, resolution):
   
    correct = True

    edge_bbox_width = 20.0/depth_linear_image.shape[1]
    edge_bbox_height = max(0, bbox[3] - 10/depth_linear_image.shape[0])

    x_adj = 5/depth_linear_image.shape[1]

    # Get the world_coords of the left of the bounding box
    if bbox[0] - bbox[2]/2 + x_adj < 0:
        correct = False
    l_bbox = np.array([bbox[0] - bbox[2]/2 + x_adj, bbox[1], edge_bbox_width, bbox[3]])
    lx, ly, lz, ldepth, ldepth_linear = get_real_coords(l_bbox, robot_yaw, camera_pos, camera_ori, depth_image, depth_linear_image, camera_intrinsic_mat, resolution)

    # Get the world_coords of the right of the bounding box
    if bbox[0] + bbox[2]/2 - x_adj > depth_linear_image.shape[1]:
        correct = False
    r_bbox = np.array([bbox[0] + bbox[2]/2 - x_adj, bbox[1], edge_bbox_width, bbox[3]])
    rx, ry, rz, rdepth, rdepth_linear = get_real_coords(r_bbox, robot_yaw, camera_pos, camera_ori, depth_image, depth_linear_image, camera_intrinsic_mat, resolution)

    # Estimate centroid -- assume (lx,ly) as reference point, (rx, ry) lies on positive x-axis of reference
    dist = resolution / 2
    ang = np.arctan2(ry-ly, rx-lx)

    n_point = np.array([dist, dist])

    #Rotate and translate
    rot = np.array([[np.cos(ang), -np.sin(ang)],
                    [np.sin(ang), np.cos(ang)]])
    n_point = np.matmul(rot, n_point) + np.array([lx,ly])

    return n_point[0], n_point[1], (lz+rz)/2.0, ldepth, ldepth_linear, correct


def obj_detection(vlm, cam_int_mat, dino_model, obj_tp, rgb_img, depth_img, config, camera_pos, robot_yaw, use_GD=True, tp='MANUAL', feature=None):
    if use_GD:
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
                prompt = f"Given the image and object type `{obj_tp}`, what is the value of the feature `{feature}`? Please respond with only a single word, answering the above question."

                response = vlm.generate(prompt, img)

                feature_ret_vals.append(response)

        return boxes, real_world_coords, feature_ret_vals, logits
    elif tp == 'MANUAL': # Manual labelled points
        def in_fov(chair_loc, camera_pos, robot_yaw, ang):
            # Get FOV boundaries
            max_angle = np.deg2rad(ang)
            min_angle = np.deg2rad(-1*ang)
            
            # Angle to chair
            delta = chair_loc - camera_pos

            yaw = np.arctan2(delta[1], delta[0]) - robot_yaw

            if yaw > min_angle and yaw < max_angle:
                return True

            return False

        #chair_locs = [
        #    np.array([1.525443434715271, 1.051008939743042, 0.45048078894615173]),
        #    np.array([1.5741584300994873, -0.18128417432308197, 0.4504806101322174]), 
        #    np.array([0.9688517451286316, 0.37503647804260254, 0.45048055052757263]),
        #    np.array([-0.40415289998054504, 1.1010076999664307, 0.48770344257354736]),
        #    np.array([-1.2552380561828613, 1.1008754968643188, 0.4877033829689026]),
        #    np.array([0.11770259588956833, -3.4406189918518066, 0.3834041953086853])
        #]
        chair_locs = [
            np.array([0.75,0,0.15]),
            np.array([0, -1, 0.15])
        ]

        all_locs = []
        #with open('/robodata/user_data/npatt/OmniGibson/mod_scenes/Rs_int_DONE.json') as f:
        #    obj_file = json.load(f)

        #for obj_name in obj_file['state']['object_registry']:
        #    all_locs.append(obj_file['state']['object_registry'][obj_name]['root_link']['pos'])

        ret_chair_locs = []
        ret_logits = []
        for chair_loc in chair_locs:
            if in_fov(chair_loc, camera_pos, robot_yaw, ang=30):
                # Check for obstruction
                dist_to_chair = np.linalg.norm(chair_loc - camera_pos)

                obstructed = False
                for loc in all_locs:
                    if in_fov(np.array(loc), camera_pos, robot_yaw, ang=30):
                        dist_to_loc = np.linalg.norm(np.array(loc) - camera_pos)

                        if dist_to_loc < dist_to_chair:
                            obstrcuted = True

                if not obstructed:
                    ret_chair_locs.append(chair_loc)
                    ret_logits.append(2)

        return ret_chair_locs, ret_logits

    elif tp == 'RWC': # Use simulator bounding boxes
        bboxes = np.array(state['robot0:eyes:Camera:0']['bbox_3d'])

        obj_ids = []
        instance_info = info['obs_info']['robot0']['robot0:eyes:Camera:0']['seg_instance']
        semantic_info = info['obs_info']['robot0']['robot0:eyes:Camera:0']['seg_semantic']
        img = np.array(state['robot0:eyes:Camera:0']['rgb'])
       
       
        """
        print(bboxes)
        c_img = colorize_bboxes_3d(bboxes, img, camera_params)
        plt.imshow(c_img)
        plt.show()
        """ 
        
        box_files = glob.glob('/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/boxes_*.npy')
        img_files = glob.glob('/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/image_*.npy')
        coords_files = glob.glob('/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/real_coords_*.npy')
       
        box_files_sorted = []
        img_files_sorted = []
        coords_files_sorted = []
        for i in range(len(img_files)):
            box_file = box_files[i]
            box_file_num = int(box_file.rstrip('.npy').lstrip('/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/boxes_'))
            box_files_sorted.append((box_file_num, box_file))
            img_file = img_files[i]
            img_file_num = int(img_file.rstrip('.npy').lstrip('/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/image_'))
            img_files_sorted.append((img_file_num, img_file))
            coords_file = coords_files[i]
            coords_file_num = int(coords_file.rstrip('.npy').lstrip('/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/real_coords_'))
            coords_files_sorted.append((coords_file_num, coords_file))

        img_files_sorted.sort()
        box_files_sorted.sort()
        coords_files_sorted.sort()

        print(img_files_sorted)

        for i in range(len(img_files_sorted)):
            box_file_num, box_file = box_files_sorted[i]
            img_file_num, img_file = img_files_sorted[i]
            coords_file_num, coords_file = coords_files_sorted[i]
            assert box_file_num == img_file_num, f'{box_file_num} != {img_file_num}'
            bbox = np.load(box_file)
            img = np.load(img_file)
            coords = np.load(coords_file)
            print(coords)
            c_img = colorize_bboxes_3d(bbox, img, camera_params)
            plt.imshow(c_img)
            plt.show()

        assert False
        

        # Instances
        for inst_id in instance_info:
            if instance_info[inst_id].find(obj_tp.lower()) != -1:
                obj_ids.append(inst_id)
        
        # Semantic
        #for sem_id in semantic_info:
        #    if semantic_info[sem_id].find(obj_tp.lower()) != -1:
        #        obj_ids.append(sem_id)


        coords = []
        logits = []
        ret_bboxes = []
        #from omni.syntheticdata.scripts.helpers import get_bbox_3d_corners
        corners_3d = get_bbox_3d_corners(bboxes)
        for obj_id in obj_ids:
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                if bbox[0] == obj_id:
                    corners = corners_3d[i, :, :]
                    #x_min = bbox[1]
                    #y_min = bbox[2]
                    #z_min = bbox[3]
                    #x_max = bbox[4]
                    #y_max = bbox[5]
                    #z_max = bbox[6]
                    #x = (x_max + x_min)/2
                    #y = (y_max + y_min)/2
                    #z = (z_max + z_min)/2
                    avg = np.mean(corners, axis=0)
                    coords.append((avg[0], avg[1], avg[2]))
                    ret_bboxes.append(bbox)

                    logits.append(2)

        return coords, logits, ret_bboxes
    else: # Use simulator semantic segmentation
        instance_segmentation = np.array(state['robot0:eyes:Camera:0']['seg_instance'])
        depth_image = np.array(state['robot0:eyes:Camera:0']['depth_linear'])

        obj_ids = []
        seg_inst_info = info['obs_info']['robot0']['robot0:eyes:Camera:0']['seg_instance']
        print("\n\nINFO:\n", seg_inst_info, "\n\n")
        for inst_id in seg_inst_info:
            if seg_inst_info[inst_id].find(obj_tp.lower()) != -1:
                obj_ids.append(inst_id)


        boxes = []
        logits = []
        for obj_id in obj_ids:
            # Make bbox (xyxy)
            y, x = np.where(instance_segmentation == obj_id)

            bbox = (np.min(x), np.min(y), np.max(x), np.max(y))

            # Check that object is fully within frame
            if bbox[0] <= 0 or bbox[1] <= 0:
                continue
            if bbox[1] >= instance_segmentation.shape[1] - 1 or bbox[3] >= instance_segmentation.shape[0] - 1:
                continue

            # Check occlusions
            min_object_depth = -1
            for x in range(bbox[0], bbox[2] + 1):
                for y in range(bbox[1], bbox[3] + 1):
                    if instance_segmentation[y,x] == obj_id:
                        if depth_image[y,x] < min_object_depth or min_object_depth == -1:
                            min_object_depth = depth_image[y,x]


            occluded_pix_count = 0
            for x in range(bbox[0], bbox[2] + 1):
                for y in range(bbox[1], bbox[3] + 1):
                    if instance_segmentation[y,x] == obj_id:
                        continue

                    if depth_image[y,x] < min_object_depth:
                        occluded_pix_count += 1

            total_pix_count = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            occluded = (occluded_pix_count / total_pix_count) >= config['observation_calc_params']['occluded_percent_to_ignore']

            if not occluded:
                cx = ((bbox[0] + bbox[2])/2) / instance_segmentation.shape[1]
                cy = ((bbox[1] + bbox[3])/2) / instance_segmentation.shape[0]
                w = (bbox[2] - bbox[0]) / instance_segmentation.shape[1]
                h = (bbox[3] - bbox[1]) / instance_segmentation.shape[0]
                cxcywh_bbox = np.array([cx, cy, w, h])

                print("OBJ DETECTION BBOX: ", cxcywh_bbox)

                boxes.append(cxcywh_bbox)

                logits.append(2)

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

    BOX_THRESHOLD = 0.6
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
            o_map_xy = obstacle_map.world2vox(np.array([x, y, 0]))

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

def check_cluster(x, y, z, belief):
    res = belief.map_params['res']
    
    # Merge close clusters
    merged_indices = []
    new_clusters = []
    for i in range(len(belief.clusters)):
        if i in merged_indices:
            continue

        cur_cluster, cur_num = belief.clusters[i]

        for j in range(i+1, len(belief.clusters)):
            if j in merged_indices:
                continue

            test_cluster, test_num = belief.clusters[j]

            if np.linalg.norm(cur_cluster-test_cluster) < 0.8 * res:
                cur_cluster = ((cur_cluster * cur_num) + (test_cluster * test_num)) / (cur_num+test_num)
                cur_num += test_num

                merged_indices.append(i)
                merged_indices.append(j)

        new_clusters.append((cur_cluster, cur_num))

    belief.clusters = new_clusters


    p = np.array([x,y,z])
    closest_dist = -1
    closest_center_idx = 0
    i = 0

    for center, number in belief.clusters:
        dist = np.linalg.norm(p-center)
        if dist < closest_dist or closest_dist == -1:
            closest_dist = dist
            closest_center_idx = i

        i += 1

    if closest_dist < res and closest_dist != -1: # Found Closest Cluster
        cur_center, cur_num = belief.clusters[closest_center_idx]
        
        new_num = cur_num + 1
        new_center = (cur_center * cur_num + p)/new_num

        belief.clusters[closest_center_idx] = (new_center, new_num)

        return new_center[0], new_center[1], new_center[2]

    else: # Make new cluster
        belief.clusters.append((p, 1))
        return p[0], p[1], p[2]

def get_vox_preds(vlm, robot_yaw, camera_pos, camera_pose, belief, obj_tp, rgb_image, depth_image, dino_model, config, obstacle_map, camera_intrinsic_mat, feature=None, iteration=0):
    """
    Function to get predicted value of existence at each voxel (for an object type) 
    give observation

    :param:


    :returns: np.array with same size as belief, where voxels within observation are updated based on 
    predicted value of object existence.
    """

    print("Robot Yaw: ", robot_yaw)

    voxel_preds = np.ones(belief.p.shape)
    voxel_preds *= -1
    
    # Make 0 in all visible voxels
    resolution = belief.map_params['res']
    size = belief.map_params['size']

    print("RESOLUTION IN GET_VOX_PRED: ", resolution)
    print("AND SIZE: ", size)

    z_dim_max = belief.z_dim
    print("Starting FOV")
    voxel_preds = get_fov_from_depth_image(camera_pos, robot_yaw, depth_image, voxel_preds, resolution, belief.map_params['z_res'], size, config, camera_intrinsic_mat)
    
    boxes, real_world_coords, feature_ret_vals, logits = obj_detection(vlm, camera_intrinsic_mat, dino_model, obj_tp, rgb_image, depth_image, config, camera_pos, robot_yaw, feature=feature)
    # Get the corresponding voxels
    found_obj = False
    if len(real_world_coords) > 0:
        found_obj = True

        #np.save(f'/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/real_coords_{iteration}.npy', real_world_coords)
        #np.save(f'/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/boxes_{iteration}.npy', bboxes)
        #np.save(f'/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/image_{iteration}.npy', np.array(state['robot0:eyes:Camera:0']['rgb']))
        #np.save(f'/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/depth_linear_image_{iteration}.npy', np.array(state['robot0:eyes:Camera:0']['depth_linear']))
        #np.save(f'/robodata/user_data/npatt/OmniGibson/debug/grounding_dino_bb_images/depth_image_{iteration}.npy', np.array(state['robot0:eyes:Camera:0']['depth']))

    print("First BOXES")
    Rotation = np.array([
        [np.cos(robot_yaw), -np.sin(robot_yaw), 0],
        [np.sin(robot_yaw), np.cos(robot_yaw), 0],
        [0,0,1]])

    feature_vox_ret_vals = []
    for i in range(len(real_world_coords)):
        x, y, z = real_world_coords[i]
        #x, y, z = np.matmul(Rotation, real_world_coords[i]) + camera_pos
        print("REAL WORLD: ", x,y,z)
        score = logits[i] #1/(1+np.exp(-1*logits[i]))
        
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
            
    
    #if found_obj:
        #np.save(f'/robodata/user_data/npatt/OmniGibson/debug/voxel_predictions/{iteration}.npy', voxel_preds)
        #np.save(f'/robodata/user_data/npatt/OmniGibson/debug/obstacle_maps/{iteration}.npy', obstacle_map.obstacles)

    return voxel_preds, feature_vox_ret_vals, found_obj
