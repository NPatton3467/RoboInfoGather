import importlib
TSDFPlanner = importlib.import_module('explore-eqa.src.tsdf').TSDFPlanner
from RoboInfoGather.program_utils import *
from RoboInfoGather.pomdp import *
from RoboInfoGather.MCTS_planner import *
from RoboInfoGather.map_utils import *
from RoboInfoGather.observation_utils import *

import logging

import time

import csv
import pickle
import math
import quaternion
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import imageio as iio
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from collections import OrderedDict

def get_sampled_point(pts, angle, cfg, pomdp, tsdf_planner, scene_data):
    best_reward = 0
    best_sample_pt_world_space = None
    best_sample_angle = None

    num_checked = 0


    while num_checked < cfg['planner_params']['max_num_samples']:
        num_checked += 1

        # Find where legal
        _, unoccupied = tsdf_planner.get_island_around_pts(pts)
        legal_locs_in_tsdf_space = np.argwhere(unoccupied)

        # Bin samples
        loc_bin = np.random.randint(0, cfg.rf_params.num_loc_bins)
        ang_bin = np.random.randint(0, cfg.rf_params.num_ang_bins)

        num_legal = legal_locs_in_tsdf_space.shape[0]
        sample_loc_min = int(loc_bin * (num_legal / cfg.rf_params.num_loc_bins))
        sample_loc_max = int(min(num_legal, (loc_bin + 1) * (num_legal / cfg.rf_params.num_loc_bins)))
        sample_loc_idx = np.random.randint(sample_loc_min, sample_loc_max)
        sample_loc_in_tsdf_space = legal_locs_in_tsdf_space[sample_loc_idx]


        sample_ang_min = int(ang_bin * (360 / cfg.rf_params.num_ang_bins))
        sample_ang_max = int(min(360, (ang_bin + 1) * (360 / cfg.rf_params.num_ang_bins)))
        sampled_angle = np.deg2rad(np.random.randint(sample_ang_min, sample_ang_max))

        # Compute reward
        reward = 0
        sample_loc_in_world_space = sample_loc_in_tsdf_space[:2] * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
        for key in pomdp.bel.keys():
            belief = pomdp.bel[key]
            reward_pt = np.array([sample_loc_in_world_space[0],
                                sample_loc_in_world_space[1],
                                pts[2]])
            reward += pomdp.reward_funcs[key].eval(belief, tsdf_planner, reward_pt, sampled_angle)

        # Check if new best
        if reward > best_reward or best_sample_pt_world_space is None:
            best_sample_pt_world_space = sample_loc_in_world_space
            best_sample_angle = sampled_angle

    fig, ax = plt.subplots()
    best_sample_pt_world_space = np.append(best_sample_pt_world_space, scene_data['floor_height']) 

    return best_sample_pt_world_space, best_sample_angle, fig


def get_perfect_next_point(tsdf_planner, scene_data):
    """
    Used for debugging. Prompts user to select coordinates in the image based on obj_tp

    Inputs:
        obj_tp:         The object type to look for in the image
        img:            The image to select coordinates in 

    Ouputs:
        coords:         A list of pixel coordinates cooresponding to the user's selection
    """

    # Overlay tsdf_volume on trav_map
    trav_map = scene_data['trav_map']
    tsdf = np.rot90(np.flip(tsdf_planner._tsdf_vol_cpu, axis=1), k=1)
    tsdf = tsdf[:,:,3]
    tsdf = cv2.resize(tsdf, (trav_map.shape[0], trav_map.shape[1]))

    fig, ax = plt.subplots()
    ax.imshow(trav_map)
    sns.heatmap(tsdf, alpha=0.4, cmap='Blues', ax=ax, cbar=False)
    plt.axis('off')
    #plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)

    root = tk.Tk()
    img = np.copy(iio.imread('temp.png'))
    img = cv2.resize(img, (trav_map.shape[0], trav_map.shape[1]))
    print(img.shape, trav_map.shape)
    os.remove('temp.png')
    app = PixelSelector(root, img)
    root.mainloop()

    pixels = np.array(app.pixels)

    # Clean up the gui
    root.destroy()
    app.shutdown()
    del(app)
    del(root)

    ############################
    # Put pixels in TSDF frame #
    ############################
    og_tsdf_shape = np.array(tsdf_planner._tsdf_vol_cpu.shape[:2])
    tsdf_shape = np.array(tsdf.shape)
    print(og_tsdf_shape)

    # Resize
    px_pt_resized = pixels[0] * og_tsdf_shape / tsdf_shape
    px_ang_resized = pixels[1] * og_tsdf_shape / tsdf_shape

    print(px_pt_resized)
    print(px_ang_resized)

    # Rotate
    """
    Seems like this is coverd by the transformations
    to get the tsdf into the image frame
    theta = np.deg2rad(-90)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
    
    px_pt_rotated = np.dot(R, px_pt_resized - og_tsdf_shape/2)
    px_pt_rotated = np.array([-px_pt_rotated[1], px_pt_rotated[0]]) # FLIP
    #px_pt_rotated = np.array([px_pt_rotated[0], -px_pt_rotated[1]]) # FLIP
    px_ang_rotated = np.dot(R, px_ang_resized - og_tsdf_shape/2)
    px_ang_rotated = np.array([-px_ang_rotated[1], px_ang_rotated[0]]) # FLIP
    #px_ang_rotated = np.array([px_ang_rotated[0], -px_ang_rotated[1]]) # FLIP

    print(px_pt_rotated)
    print(px_ang_rotated)

    # Translate back to final tsdf frame
    tsdf_pt = px_pt_rotated + og_tsdf_shape/2
    tsdf_ang = px_ang_rotated + og_tsdf_shape/2

    print(tsdf_pt)
    print(tsdf_ang)
    """

    tsdf_pt = px_pt_resized
    tsdf_ang = px_ang_resized

    # Plot to check
    c_tsdf = np.copy(tsdf_planner._tsdf_vol_cpu)
    c_tsdf[int(tsdf_pt[0]), int(tsdf_pt[1]), 3] = 5
    c_tsdf[int(tsdf_ang[0]), int(tsdf_ang[1]), 3] = 5

    #plt.imshow(c_tsdf[:,:,3])
    #plt.show()

    
    # Get world coordinates 
    vox_coords = np.array([[tsdf_pt[0], tsdf_pt[1], 0]])
    world_pt = tsdf_planner.vox2world(tsdf_planner._vol_origin, vox_coords, tsdf_planner._voxel_size)[0]

    print(world_pt)

    # Get angle
    diff = tsdf_ang - tsdf_pt
    angle = np.arctan2(diff[1], diff[0])

    print(angle)

    pts_normal = np.append(world_pt[:2], scene_data['floor_height'])
    print(pts_normal)

    return pts_normal, angle, fig

# Complile the information produced into the result to be saved to a file
def get_result(
        prog,
        symbolic_info,
        question,
        pomdp,
        text_answer,
        cum_sim_score,
        cfg,
        question_ind,
        scene_data
    ):

    result = {}

    # Execute Symbolic Info
    exec_prog = copy.deepcopy(prog)
    query_exec_res = exec_prog.execute(symbolic_info)
    print("Prog Result:\n", query_exec_res)
    result['query_execution_result'] = query_exec_res
    result['full_symb_info'] = symbolic_info
    result['question'] = question
    result['pomdp'] = pomdp
    result['prog'] = prog

    # Get LLM to generate Natural Lanugage answer
    query_str = prog.pretty_str()
    nl_ans = get_nl_answer(query_exec_res, question, query_str)
    print("Natural Language Answer:\n", nl_ans)
    result['output_answer'] = nl_ans
    result['True_Answer'] = text_answer

    # Increment LLM similarity score
    print("Actual Answer: ", text_answer)
    raw_sim_score = eval_similarity(question, nl_ans, text_answer)
    result['Raw_Similarity_Score'] = raw_sim_score
    # Extract Int Score
    try:
        sim_score = int(raw_sim_score)
    except Exception as e:
        print(f"Failed to make int from model output: {str(e)}")
        sim_score = 1
    result['Similarity_Score'] = sim_score

    cum_sim_score += (sim_score - 1)/4
    cur_llm_match_percent = (cum_sim_score/(question_ind+1)) * 100
    result['current_cum_sim_score'] = cum_sim_score
    result['current_llm_match_percent'] = cur_llm_match_percent
    
    print("Current Cumulative Sim Score: ", cum_sim_score)
    print("Current LLM-Match %: ", cur_llm_match_percent)

    # Episode summary
    logging.info(f"\n== Episode Summary")
    logging.info(f"Scene: {scene_data['scene_name']}")
    logging.info(f"Question:\n{question}\nAnswer: {text_answer}")

    return result

# Call synthesis module to generate the program to be executed
# After program is generated, instantiate the POMDP from the program
def gen_program_and_pomdp(cfg, tsdf_bnds, tsdf_planner, question, pos, angle, max_attempts=10000):

    """
    Generate the program and corresponding POMDP to be used for the given task

    Inputs:
        cfg: The configuration file for experiment specific configurations
        tsdf_bnds: the bounds of the map used by the tsdf_planner
        tsdf_planner: TSDFPlanner used throughout execution of experiments
                        needed here for getting additional information such as
                        voxel sizes
        question: The natural language question for this task. Used for generating the program
        pos: Position of the robot, np.array of [x, y, z] in simulator map frame
        angle: Yaw of the robot in radians, in simulator map frame
        max_attempts: Number of retries on synthesizing a valid program before giving up

    Outputs:
        prog: The program generated that is a valid within the DSL defined in dsl.py
        pomdp: An instance of POMDP as defined in pomdp.py. This contains beliefs for 
                each object type and their respective features of interest.
    """

    attempts = 0
    prog = None
    pomdp = None
    resolution = cfg.tsdf_grid_size
    size = int(max(abs(tsdf_bnds[0][0] - tsdf_bnds[0][1]), abs(tsdf_bnds[1][0] - tsdf_bnds[1][1])) / resolution)

    print("SIZE: ", size)
    print("RESOLUTION: ", resolution)
    print(tsdf_bnds)
    print("VOL DIM: ", tsdf_planner._vol_dim)
    print("VOX SIZE: ", tsdf_planner._voxel_size)
    
    #This format: np.array([-way_point.loc.y, pts[1], way_point.loc.x])
    # Wrapped in try block since pomdp generation can fail through assertions
    # if the output from the LLM synthesizer is incorrect
    while attempts < 10000:
        attempts += 1
        try:
            # Call synthesizer
            prog = gen_prog_from_nl(question)

            # Generate the pomdp from the program
            if type(prog) is Prog:
                pomdp = gen_pomdp_from_query(
                        query=prog.expressions[0],
                        pos=pos,
                        yaw=angle,
                        trav_map_og_dim=tsdf_planner._vol_dim,
                        trav_map_og_res=resolution,
                        vol_origin=tsdf_planner._vol_origin,
                        configs=cfg
                    )
            else:
                pomdp = gen_pomdp_from_query(
                        query=prog,
                        pos=pos,
                        yaw=angle,
                        trav_map_og_dim=tsdf_planner._vol_dim,
                        trav_map_og_res=resolution,
                        vol_origin=tsdf_planner._vol_origin,
                        configs=cfg
                    )
            break
        except Exception as e:
            print(f"Could not generate program, attempt {attempts}\n{str(e)}")


    return prog, pomdp

# Take a step in the environment and recieve updated state (rgb, depth)
def env_update(cnt_step, pts, pitch, roll, angle, env):

    """
    This function updates the environment for one step

    Inputs:
        cnt_step:   The current step of simulation
        pts:        The current robot position in the simulator map frame
        pitch:      The camera pitch in the simulator map frame
        roll:       The camera roll in the simulator map frame
        angle:      The camera/robot yaw in the simulator map frame
        env:        The simulation environment to take a step in

    Outputs:
        pts:            The new robot position in the simulator map frame
        angle:          The new camera/robot yaw in the simulator map frame
        cam_pose:       The new rotation matrix + translation of the camera in the simulation map frame
        cam_pose_tsdf:  The new rotation matrix + translation of the camera in the tsdf map frame
        camera_pos:     The new position of the camera in the simulation map frame
        rgb:            The new rgb observation
        depth:          The new depth observation
    """

    # Save step info and set current pose
    logging.info(f"Current pts: {pts}")
    # TEMP JUST ROTATE
    #ori_to_send = Rotation.from_euler('xyz', [pitch, roll, angle+np.deg2rad(90)], degrees=False).as_quat()
    ori_to_send = Rotation.from_euler('xyz', [pitch, roll, angle], degrees=False).as_quat()
    while np.isnan(ori_to_send).any():
        angle += 1
        ori_to_send = Rotation.from_euler('xyz', [pitch, roll, angle], degrees=False).as_quat()

    print("Setting rotation")
    env.robots[0].set_position_orientation(pts, ori_to_send)
    action = OrderedDict([('rob', np.array([0 , 0]))])
    print("Sending sync actions")
    state, _, _, _, info = env.step(action) # Take Empty step to get observations
    state, _, _, _, info = env.step(action) # Take Empty step to get observations
    state, _, _, _, info = env.step(action) # Take Empty step to get observations
    print("Getting new position")
    pts, _ = env.robots[0].get_position_orientation()
    pts = np.array(pts.cpu().detach())
    print("Getting new RPY")
    roll, pitch, angle = env.robots[0].get_rpy()
    angle = angle.cpu().detach()
    pts_normal = pts

    # Update camera info
    print("Getting camera info")
    translation_0, quaternion_0 = env.robots[0]._sensors['rob:eyes:Camera:0'].get_position_orientation()
    roll_cam, pitch_cam, _ = env.robots[0]._sensors['rob:eyes:Camera:0'].get_rpy()
    quaternion_0 = np.array(quaternion_0.detach().cpu()).astype(float)
    translation_0 = np.array(translation_0.detach().cpu()).astype(float)
    cam_pose = np.eye(4)
    print("Quaternion Shape: ", quaternion_0)
    print("Camera RPY: ", roll_cam, pitch_cam, angle)
    cam_pose[:3, :3] = Rotation.from_euler('xyz', [0, 0, angle]).as_matrix()
    cam_pose[:3, 3] = translation_0

    cam_pose_tsdf = np.eye(4)
    cam_pose_tsdf[:3, :3] = Rotation.from_euler('yxz', [pitch_cam, roll_cam-np.deg2rad(180), angle-np.deg2rad(90)]).as_matrix()
    cam_pose_tsdf[:3, 3] = translation_0

    print("\n\nCAMERA STUFF:")
    camera_pos = translation_0
    print("translation_0 (sensor.position): ", translation_0)
    print("cam_pose: ", cam_pose)
    print("cam_pose_tsdf: ", cam_pose_tsdf)
    print("camera_pos: ", camera_pos)
    print("pts: ", pts)
    print("pts_normal: ", pts_normal)
    print("\n\n")

    # Get observation at current pose
    print("Getting new observations")
    rgb = np.array(state['rob']['rob:eyes:Camera:0']['rgb'].detach().cpu())
    depth = np.array(state['rob']['rob:eyes:Camera:0']['depth_linear'].detach().cpu())
    bbox_3d = state['rob']['rob:eyes:Camera:0']['bbox_3d']
    seg_sem = state['rob']['rob:eyes:Camera:0']['seg_semantic']
    seg_inst = state['rob']['rob:eyes:Camera:0']['seg_instance']
    seg_inst_id = state['rob']['rob:eyes:Camera:0']['seg_instance_id']

    obs = {
            "rgb": rgb,
            "depth": depth,
            "bbox_3d": bbox_3d,
            "seg_sem": seg_sem,
            "seg_inst": seg_inst,
            "info": info
        }

    return pts, angle, cam_pose, cam_pose_tsdf, camera_pos, obs

# Get frontier points from TSDF volume and see if there are any in the image
# to check if they are good next waypoints with VLM
def setup_frontier_pts_in_image(
        pts_normal,
        cam_pose_tsdf,
        tsdf_planner,
        img_width,
        img_height,
        cam_intr,
        cfg,
        episode_data_dir,
        cnt_step
    ):

    """
    Use the tsdf planner to find the prompt points within view of the current image.
    Return the pixel value of these prompt_points

    Inputs:
        pts_normal:             The robot position in the simulator map frame
        cam_pose_tsdf:          The camera rotation + translation matrix in the tsdf map frame
        tsdf_planner:           Instance of TSDFPlanner used in for the current task
        img_width:              The width in pixels of the current image
        img_height:             The height in pixels of the current image
        cam_intr:               The camera intrinsic matrix, used to project pixel frame <-> simulator map frame
        cfg:                    The current task config data
        episode_data_dir:       The directory to save task information to
        cnt_step:               The current simulator step

    Outputs:
        prompt_points_pix:      The pixel values of any points of interest within the current field of view
    """

    print("PTS NORMAL: ", pts_normal)
    print("CAM POSE TSDF: ", cam_pose_tsdf)

    prompt_points_pix, fig = (
        tsdf_planner.find_prompt_points_within_view(
            pts_normal,
            img_width,
            img_height,
            cam_intr,
            cam_pose_tsdf,
            **cfg.visual_prompt,
        )
    )
    fig.tight_layout()
    #plt.savefig(
    #    os.path.join(
    #        episode_data_dir, "{}_prompt_points.png".format(cnt_step)
    #    )
    #)
    plt.close()

    return prompt_points_pix

# Draw the points on the current image for the VLM to rank 
def draw_image_pts(rgb_im, prompt_points_pix, cfg, episode_data_dir, cnt_step):

    """
    Draws the pixel points on the current image, with letters (A through D) for the VLM to rank

    Inputs:
        rgb_im:                 The current RGB image observation
        prompt_points_pix:      The pixel points of interest (where to draw the letters)
        cfg:                    The current task configuration data
        episode_data_dir:       Directory to store current task information
        cnt_step:               The current simulator step

    Outputs:
        rgb_im_draw:            rgb_im with the points of interest drawn with A through D
    """

    draw_letters = ["A", "B", "C", "D"]  # always four
    fnt = ImageFont.truetype(
        "RoboInfoGather/explore-eqa/data/Open_Sans/static/OpenSans-Regular.ttf",
        30,
    )
    rgb_im_draw = rgb_im.copy()
    draw = ImageDraw.Draw(rgb_im_draw)
    for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
        draw.ellipse(
            (
                point_pix[0] - cfg.visual_prompt.circle_radius,
                point_pix[1] - cfg.visual_prompt.circle_radius,
                point_pix[0] + cfg.visual_prompt.circle_radius,
                point_pix[1] + cfg.visual_prompt.circle_radius,
            ),
            fill=(200, 200, 200, 255),
            outline=(0, 0, 0, 255),
            width=3,
        )
        draw.text(
            tuple(point_pix.astype(int).tolist()),
            draw_letters[prompt_point_ind],
            font=fnt,
            fill=(0, 0, 0, 255),
            anchor="mm",
            font_size=12,
        )
    #rgb_im_draw.save(
    #    os.path.join(episode_data_dir, f"{cnt_step}_draw.png")
    #)

    return rgb_im_draw

# Get the reward value of the waypoints suggested in the image from the belief
def get_reward_for_pix(
        depth,
        px,
        py,
        pts,
        angle,
        camera_pos,
        cam_pose,
        cam_intr,
        cfg,
        tsdf_planner,
        pomdp
    ):

    """
    Get the reward value of the waypoints suggested in the image, based on infromation gain in the belief

    Inputs:
        depth:          The current DEPTH image observation
        px:             The pixel's x value (int)
        py:             The pixel's y value (int)
        pts:            The robot's current position in the simulator map frame
        angle:          The robot/camera current yaw in the simulator map frame
        camera_pos:     The camera position in the simulator map frame
        cam_pose:       The camera rotation + translation matrix in the simulator map frame
        cam_intr:       The camera intrinsic matrix, used for projecting (pixel frame <-> simulator map frame)
        cfg:            The current task configuration data
        tsdf_planner:   Instance of TSDFPlanner, used for selecting next waypoints
        pomdp:          Instance of POMDP, used for mainting beliefs, for generating information gain reward

    Outputs:
        reward:         The reward calculated for the proposed pixel. This reward is the information gained
                            by making an observation at that point (in the simulator map frame)
    """

    cur_depth = depth[py,px]
    print("Getting Reward")
    world_coords = get_world_coords_from_depth(px, py, cur_depth, camera_pos, cam_pose, cam_intr)
    node = MCTS_Tree_Node(
            loc=Loc(world_coords[0], world_coords[1], angle),
            obstacle_map = tsdf_planner,
            num_prev_obs = 0,
            max_obs = 1,
            config = cfg,
            inbound_act = Action.OBS
            )
    root = MCTS_Tree_Node(
            loc=Loc(pts[2], -pts[0], angle),
            obstacle_map = tsdf_planner,
            num_prev_obs = 0,
            max_obs = 1,
            config = cfg,
            inbound_act = Action.OBS
            )
    reward = 0
    for key in pomdp.bel.keys():
        belief = pomdp.bel[key]
        reward += pomdp.reward_funcs[key].eval(belief, tsdf_planner, root, node)

    return reward

# Use VLM to get local semantic values (lsv) and integrate that with the TSDF
# volume semantic values to rank frontiers
def integrate_vlm_loss(
        prompt_points_pix,
        vlm,
        rgb_im_draw,
        prompt_lsv,
        draw_letters,
        depth,
        pts,
        angle,
        camera_pos,
        cam_pose,
        cam_intr,
        cfg,
        tsdf_planner,
        pomdp
    ):

    """
    For each prompt pixel, get the VLM loss and use this as the local semantic value.
    Integrate the local semantic value into the TSDF planner for use in waypoint selection

    Inputs:
        prompt_points_pix:      The pixel locations of potential next way points for the VLM to rank
        vlm:                    The Visual Language Model (prismatic) used to rank the prompt points
        rgb_im_draw:            The current RGB obsevation with the prompt points drawn on
        prompt_lsv:             The prompt used to query the VLM for ranking the points in the image
        draw_letters:           The letters used for the prompt points (A through D)
        depth:                  The current DEPTH image observation
        pts:                    The current robot position in the simulator map frame
        angle:                  The current robot/camera yaw in the simulator map frame
        camera_pos:             The current camera position in the simulator map frame
        cam_pose:               The current camera rotation + translation matrix in the simulator map frame
        cam_intr:               The camera intrinsic matrix used for projecting pixel-frame <-> simulator map frame
        cfg:                    The current task configuration data
        tsdf_planner:           Instance of TSDFPlanner used to find next points
        pomdp:                  Instance of POMDP built from the current task's program.
                                    Used for generating reward to augment local semantic value with information gain

    Outputs:
    """

    actual_num_prompt_points = len(prompt_points_pix)
    if actual_num_prompt_points >= 1:

        # logging.info(f"Prompt Exp: {prompt_text}")
        #lsv = np.ones(actual_num_prompt_points)*0.75
        lsv = vlm.get_loss(
            rgb_im_draw,
            prompt_lsv,
            draw_letters[:actual_num_prompt_points],
        )
        lsv *= actual_num_prompt_points / 3

        print("Got VLM loss")

        # For each prompt point -- add the info gain reward to lsv
        print("Starting semantic integration")
        for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
            py = prompt_points_pix[prompt_point_ind][1]
            px = prompt_points_pix[prompt_point_ind][0]

            if py < 0 or py >= depth.shape[0] or px < 0 or px >= depth.shape[1]:
                continue

            reward = get_reward_for_pix(
                    depth,
                    px,
                    py,
                    pts,
                    angle,
                    camera_pos,
                    cam_pose,
                    cam_intr,
                    cfg,
                    tsdf_planner,
                    pomdp
                )

            # TEMP -- No reward
            #lsv[prompt_point_ind] += 2*reward
            # END TEMP

        # Integrate semantics only if there is any prompted point
        tsdf_planner.integrate_sem(
            sem_pix=lsv,
            radius=1.0,
            obs_weight=1.0,
        )  # voxel locations already saved in tsdf class

        print("Finishing semantic integration")

# Perfrom next point prediction
# Rank with VLMs then use TSDF semantic values to get next frontier
def get_next_point(
        rgb,
        question,
        pts_normal,
        cam_pose_tsdf,
        camera_pos,
        cam_pose,
        tsdf_planner,
        img_width,
        img_height,
        cam_intr,
        cfg,
        episode_data_dir,
        cnt_step,
        vlm,
        depth,
        pts,
        angle,
        pomdp,
        scene_data
    ):

    """
    Use the VLM to rank potential waypoints within the current view. Then integrate these points in the
    TSDF planner. Then use the TSDF planner to select the next waypoint

    Inputs:
        rgb:                The current RGB image observation
        question:           The natural language question of the current task
        pts_normal:         The robot's position in the simulator map frame
        cam_pose_tsdf:      The camera's rotation + translation matrix in the TSDF map frame
        camera_pos:         The camera's translation in the simulator map frame
        cam_pose:           The camera's rotation + translation matrix in the simulator map frame
        tsdf_planner:       The TSDF planner used to select the next waypoint
        img_width:          The width of the image in pixels
        img_height:         The height of the image in pixels
        cam_intr:           The camera intrinsic matrix, used to project from the pixel coordinates to
                                simulator map frame coordinates
        cfg:                The task configuration data
        episode_data_dir:   The directory for saving task data
        cnt_step:           The current simulator step
        vlm:                The Visual Language Model (prismatic) used to rank potential waypoints in from
                                the current view
        depth:              The current DEPTH image observation
        angle:              The current robot/camera yaw in the simulator map frame
        pomdp:              The pomdp built from the current task's program.
                                Used to add info-gain reward to potential waypoints
        scene_data:         Information about the scene such as floor height, traversability

    Outputs:
        pts_normal:     The new points in the simulator map frame selected as the next waypoint
        angle:          The new robot/camera yaw in the simulator map frame selected for the next waypoint
        fig:            ??
    """

    if cfg['use_perfect_next_point']:
        return get_perfect_next_point(tsdf_planner, scene_data)
    elif cfg['no_tsdf_vlm']:
        # get the new camera position
        if cfg['use_sampling']:
            return get_sampled_point(camera_pos, angle, cfg, pomdp, tsdf_planner, scene_data)

        cam_pts_normal, angle, _, fig = tsdf_planner.find_next_pose(
            pts=camera_pos,
            angle=angle,
            flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
            debug_f_path=scene_data['debug_f_path'],
            cnt_step=cnt_step,
            **cfg.planner,
        )

        # Return the robot position
        #pts_normal = np.append(cam_pts_normal, floor_height) - cam_robot_diff 
        pts_normal = np.append(cam_pts_normal, scene_data['floor_height']) 

        #plt.show()

        return pts_normal, angle + np.deg2rad(90), fig

    # Get VLM prediction
    rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
    prompt_question = (
        question
        + "\nAnswer with the option's letter from the given choices directly."
    )
    
    # Get frontier candidates
    prompt_points_pix = []
    if cfg.use_active:
        prompt_points_pix = setup_frontier_pts_in_image(
                pts_normal,
                cam_pose_tsdf,
                tsdf_planner,
                img_width,
                img_height,
                cam_intr,
                cfg,
                episode_data_dir,
                cnt_step
            )

    # Visual prompting
    rgb_im_draw = draw_image_pts(rgb_im, prompt_points_pix, cfg, episode_data_dir, cnt_step)

    prompt_lsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Please answer with a single letter."
    # logging.info(f"Prompt Exp: {prompt_text}")

    print("Done VLM prompt set up")

    # Integrate VLM outputs with TSDF
    draw_letters = ["A", "B", "C", "D"]  # always four
    integrate_vlm_loss(
            prompt_points_pix,
            vlm,
            rgb_im_draw,
            prompt_lsv,
            draw_letters,
            depth,
            pts,
            angle,
            camera_pos,
            cam_pose,
            cam_intr,
            cfg,
            tsdf_planner,
            pomdp
        )

    print("Getting next pose")
    # Get next pose
    cam_robot_diff = camera_pos - pts_normal

    # get the new camera position
    cam_pts_normal, angle, _, fig = tsdf_planner.find_next_pose(
        pts=camera_pos,
        angle=angle,
        flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
        **cfg.planner,
    )

    # Return the robot position
    #pts_normal = np.append(cam_pts_normal, floor_height) - cam_robot_diff 
    pts_normal = np.append(cam_pts_normal, scene_data['floor_height']) 

    return pts_normal, angle, fig

# Main function of Info Gathering
# 1. Generate Program and POMDP
# 2. Loop through until max steps taken or enough information found
# 2a. Update POMDP
# 2b. Get new observations
# 2c. Use observations to get next point
def info_gather_runner(
        cfg,
        env,
        camera_data,
        scene_data,
        task_info,
        cum_sim_score,
        cnt_data,
        position_data,
        vlm_models,
        question_ind
    ):

    start_time = time.time()

    ################################
    # Set Up Planner from Question #
    ################################

    # Initialize TSDF
    tsdf_planner = TSDFPlanner(
        vol_bnds=scene_data['tsdf_bnds'],
        voxel_size=cfg.tsdf_grid_size,
        floor_height_offset=0,
        pts_init=position_data['pts'],
        init_clearance=0.5,
    )

    # Init point cloud
    pt_cloud = []
    
    # Generate program and pomdp
    prog, pomdp = gen_program_and_pomdp(
            cfg,
            scene_data['tsdf_bnds'],
            tsdf_planner,
            task_info['question'],
            position_data['pts'],
            position_data['angle']
        )
    if prog == None:
        return None

    print("\n\nProgram: ")
    print(prog.pretty_str())

    print("\n\nPOMDP Objects:\n", pomdp.bel.keys())

    print("\n\nPOMDP Features:\n")
    for obj_tp in pomdp.bel.keys():
        features = pomdp.bel[obj_tp].feature_bels.keys()
        print(f"Obj: {obj_tp}, Features: {features}")

    # Run steps
    cnt_step = 0
    num_black_in_a_row = 0
    responses = []
    result = {}
    print("Starting While Loop")
    pts = position_data['pts']

    angle = position_data['angle']
    pitch = position_data['pitch']
    roll = position_data['roll']
    while cnt_step < scene_data['num_step']:
        logging.info(f"\n== step: {cnt_step}")
        # Update environment
        print("Starting Env Update")
        pts, angle, cam_pose, cam_pose_tsdf, camera_pos, obs = env_update(
                cnt_step,
                pts,
                pitch,
                roll,
                angle,
                env
            )
        print("Done Env Update")
        step_name = f"step_{cnt_step}"
        result[step_name] = {"pts": pts, "angle": angle}

        # Update pt_cloud
        #pt_cloud.append(get_new_points(obs['depth'], camera_pos, cam_pose, camera_data['cam_intr']))

        # TSDF fusion
        print("Starting TSDF integration")
        tsdf_planner.integrate(
            color_im=obs['rgb'],
            depth_im=obs['depth'],
            cam_intr=camera_data['cam_intr'],
            cam_pose=cam_pose_tsdf,
            obs_weight=1.0,
            margin_h=int(cfg.margin_h_ratio * camera_data['img_data']['h']),
            margin_w=int(cfg.margin_w_ratio * camera_data['img_data']['w']),
        )
        print("Done TSDF Integration")

        # Save volume for debuging
        t_vol = tsdf_planner._tsdf_vol_cpu
        debug_f_path =  scene_data['debug_f_path']
        #np.save(debug_f_path+f"tsdf_volume_{cnt_step}.npy", t_vol)
        #with open(debug_f_path+f'tsdf_planner_{cnt_step}.pkl', 'wb') as f:
        #    pickle.dump(tsdf_planner, f)

        # Save pointcloud for debugging
        #np.save(debug_f_path+f"point_cloud_{cnt_step}.npy", pt_cloud)

        np.save(debug_f_path+f"img_{cnt_step}.npy", obs['rgb'])
        np.save(debug_f_path+f"depth_img_{cnt_step}.npy", obs['depth'])
        with open(debug_f_path+f"bbox_3d_{cnt_step}.pkl", 'wb') as f:
            pickle.dump(obs['bbox_3d'], f)
        np.save(debug_f_path+f"seg_inst_{cnt_step}.npy", np.array(obs['seg_inst'].detach().cpu()))
        np.save(debug_f_path+f"seg_semantic_{cnt_step}.npy", np.array(obs['seg_sem'].detach().cpu()))
        np.save(debug_f_path+f"info_{cnt_step}.npy", obs['info'])

        #################
        # Update Belief #
        #################
        print("Starting Belief Update")
        pomdp.update(
                vlm_models['vlm'],
                vlm_models['molmo_tools'],
                angle,
                camera_pos,
                cam_pose,
                obs,
                cfg,
                tsdf_planner,
                camera_data['cam_intr'],
                cnt_step,
                pts,
                debug_f_path
            )

        print("Done Belief Update")

        ########################
        # Determine next point #
        ########################
        print("Starting next point selection")
        pts, angle, fig = get_next_point(
                obs['rgb'],
                task_info['question'],
                pts,
                cam_pose_tsdf,
                camera_pos,
                cam_pose,
                tsdf_planner,
                camera_data['img_data']['w'],
                camera_data['img_data']['h'],
                camera_data['cam_intr'],
                cfg,
                task_info['episode_data_dir'],
                cnt_step,
                vlm_models['vlm'],
                obs['depth'],
                pts,
                angle,
                pomdp,
                scene_data
            )

        print("Done next point selection")

        fig.tight_layout()
        #plt.savefig(
        #    os.path.join(
        #        debug_f_path, "{}_frontier_selection.png".format(cnt_step)
        #    )
        #)
        plt.close()

        ##############################
        # Check for early completion #
        ##############################
        done, symbolic_info = pomdp.enough_info(cnt_step)

        if done:
            break

        cnt_step += 1
        print("Done Iteration: ", cnt_step)
        print("Current Symbolic Info:\n", symbolic_info)

    # Get answer from POMDP
    # Check if success using weighted prediction
    _, symbolic_info = pomdp.enough_info(cnt_step)
    print("Done")
    print("Symbolic Info: ", symbolic_info)

    #Set up result and return
    result_ans = get_result(
            prog,
            symbolic_info,
            task_info['question'],
            pomdp,
            task_info['text_answer'],
            cum_sim_score,
            cfg,
            question_ind,
            scene_data
        )

    result = {**result, **result_ans}
    
    end_time = time.time()
    result['wall_time'] = end_time - start_time
    result['tsdf_planner'] = tsdf_planner

    # Save result
    with open(
        os.path.join(cfg.output_dir, f"results_{cnt_data}.pkl"), "wb"
    ) as f:
        pickle.dump(result, f)

    return result
