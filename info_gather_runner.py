import importlib
TSDFPlanner = importlib.import_module('explore-eqa.src.tsdf').TSDFPlanner
from RoboInfoGather.program_utils import *
from RoboInfoGather.pomdp import *
from RoboInfoGather.MCTS_planner import *
from RoboInfoGather.map_utils import *

import logging

import csv
import pickle
import logging
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from collections import OrderedDict

# Complile the information produced into the result to be saved to a file
def get_result(
        prog,
        symbolic_info,
        question,
        pomdp,
        text_answer,
        cum_sim_score,
        cfg
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
    logging.info(f"Scene: {scene}, Floor: {floor}")
    logging.info(f"Question:\n{question}\nAnswer: {answer}")

    # Save data
    with open(
        os.path.join(cfg.output_dir, f"results_{cnt_data}.pkl"), "wb"
    ) as f:
        pickle.dump(result, f)

    return result

# Call synthesis module to generate the program to be executed
# After program is generated, instantiate the POMDP from the program
def gen_program(cfg, tsdf_bnds, tsdf_planner, question, pos, angle, RIG_config, max_attempts=10000):
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
                        configs=RIG_config
                    )
            else:
                pomdp = gen_pomdp_from_query(
                        query=prog,
                        pos=pos,
                        yaw=angle,
                        trav_map_og_dim=tsdf_planner._vol_dim,
                        trav_map_og_res=resolution,
                        vol_origin=tsdf_planner._vol_origin,
                        configs=RIG_config
                    )
            break
        except Exception as e:
            print(f"Could not generate program, attempt {attempts}\n{str(e)}")


    return prog, pomdp

# Take a step in the environment and recieve updated state (rgb, depth)
def env_update(cnt_step, pts, pitch, roll, angle, env):
    # Save step info and set current pose
    logging.info(f"Current pts: {pts}")
    ori_to_send = Rotation.from_euler('xyz', [pitch, roll, angle], degrees=False).as_quat()
    env.robots[0].set_position_orientation(pts, ori_to_send)
    pts_normal = pts


    # Update camera info
    translation_0, quaternion_0 = env.robots[0]._sensors['rob:eyes:Camera:0'].get_position_orientation()
    quaternion_0 = np.array(quaternion_0.detach().cpu()).astype(float)
    translation_0 = np.array(translation_0.detach().cpu()).astype(float)
    cam_pose = np.eye(4)
    print("Quaternion Shape: ", quaternion_0)
    cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion.as_quat_array(quaternion_0))
    cam_pose[:3, 3] = translation_0
    cam_pose_normal = cam_pose
    cam_pose_tsdf = np.dot(
            cam_pose_normal, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))

    print("\n\nCAMERA STUFF:")
    camera_pos = translation_0
    print("translation_0 (sensor.position): ", translation_0)
    print("cam_pose: ", cam_pose)
    print("cam_pose_normal: ", cam_pose_normal)
    print("cam_pose_tsdf: ", cam_pose_tsdf)
    print("camera_pos: ", camera_pos)
    print("pts: ", pts)
    print("pts_normal: ", pts_normal)
    print("\n\n")

    # Get observation at current pose
    action = OrderedDict([('rob', np.array([0 , 0]))])
    state, _, _, _, info = env.step(action) # Take Empty step to get observations
    rgb = np.array(state['rob']['rob:eyes:Camera:0']['rgb'].detach().cpu())
    depth = np.array(state['rob']['rob:eyes:Camera:0']['depth_linear'].detach().cpu())

    return pts, angle, cam_pose, cam_pose_tsdf, cam_pose_normal, camera_pos, rgb, depth

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
    plt.savefig(
        os.path.join(
            episode_data_dir, "{}_prompt_points.png".format(cnt_step)
        )
    )
    plt.close()

    return prompt_points_pix

# Draw the points on the current image for the VLM to rank 
def draw_image_pts(rgb_im, prompt_points_pix, cfg, episode_data_dir, cnt_step):
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
    rgb_im_draw.save(
        os.path.join(episode_data_dir, f"{cnt_step}_draw.png")
    )

    return rgb_im_draw

# Get the reward value of the waypoints suggested in the image from the belief
def get_reward_for_pix(
        depth,
        px,
        py,
        pts,
        angle,
        camera_pos,
        cam_pose_normal,
        cam_intr,
        RIG_config,
        tsdf_planner,
        pomdp
    ):

    cur_depth = depth[py,px]
    print("Getting Reward")
    world_coords = get_world_coords_from_depth(px, py, cur_depth, camera_pos, cam_pose_normal, cam_intr)
    node = MCTS_Tree_Node(
            loc=Loc(world_coords[0], world_coords[1], angle),
            obstacle_map = tsdf_planner,
            num_prev_obs = 0,
            max_obs = 1,
            config = RIG_config,
            inbound_act = Action.OBS
            )
    root = MCTS_Tree_Node(
            loc=Loc(pts[2], -pts[0], angle),
            obstacle_map = tsdf_planner,
            num_prev_obs = 0,
            max_obs = 1,
            config = RIG_config,
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
        cam_pose_normal,
        cam_intr,
        RIG_config,
        tsdf_planner,
        pomdp
    ):

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
                    cam_pose_normal,
                    cam_intr,
                    RIG_config,
                    tsdf_planner,
                    pomdp
                )
            lsv[prompt_point_ind] += 2*reward

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
        cam_pose_normal,
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
        RIG_config,
        pomdp
    ):

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
            cam_pose_normal,
            cam_intr,
            RIG_config,
            tsdf_planner,
            pomdp
        )

    print("Getting next pose")
    # Get next pose
    pts_normal, angle, pts_pix, fig = tsdf_planner.find_next_pose(
        pts=pts_normal,
        angle=angle,
        flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
        **cfg.planner,
    )

    return pts_normal, angle, pts_pix, fig

# Main function of Info Gathering
# 1. Generate Program and POMDP
# 2. Loop through until max steps taken or enough information found
# 2a. Update POMDP
# 2b. Get new observations
# 2c. Use observations to get next point
def info_gather_runner(
        cfg,
        tsdf_bnds,
        question,
        text_answer,
        cum_sim_score,
        pos,
        angle,
        pts,
        pts_normal,
        RIG_config,
        cnt_data,
        num_step,
        pitch,
        roll,
        env,
        img_width,
        img_height,
        cam_intr,
        vlm,
        molmo_tools,
        debug_f_path,
        episode_data_dir,
        floor_height
    ):

    ################################
    # Set Up Planner from Question #
    ################################

    # Initialize TSDF
    tsdf_planner = TSDFPlanner(
        vol_bnds=tsdf_bnds,
        voxel_size=cfg.tsdf_grid_size,
        floor_height_offset=0,
        pts_init=pts,
        init_clearance=cfg.init_clearance * 2,
    )
    
    # Generate program and pomdp
    prog, pomdp = gen_program(cfg, tsdf_bnds, tsdf_planner, question, pos, angle, RIG_config)
    if prog == None:
        return None

    print("\n\nProgram: ")
    print(prog.pretty_str())
    
    
    print("\n\nQuery")
    print(pomdp.query.pretty_str())

    # Run steps
    pts_pixs = np.empty((0, 2))  # for plotting path on the image
    cnt_step = 0
    num_black_in_a_row = 0
    responses = []
    result = {}
    print("Starting While Loop")
    while cnt_step < num_step:
        logging.info(f"\n== step: {cnt_step}")
        # Update environment
        pts, angle, cam_pose, cam_pose_tsdf, cam_pose_normal, camera_pos, rgb, depth = env_update(cnt_step, pts, pitch, roll, angle, env)
        step_name = f"step_{cnt_step}"
        result[step_name] = {"pts": pts, "angle": angle}

        # TSDF fusion
        tsdf_planner.integrate(
            color_im=rgb,
            depth_im=depth,
            cam_intr=cam_intr,
            cam_pose=cam_pose_tsdf,
            obs_weight=1.0,
            margin_h=int(cfg.margin_h_ratio * img_height),
            margin_w=int(cfg.margin_w_ratio * img_width),
        )

        # Save volume for debuging
        t_vol = tsdf_planner._tsdf_vol_cpu
        np.save(debug_f_path+f"tsdf_volume_{cnt_step}.npy", t_vol)

        #################
        # Update Belief #
        #################
        real_world_coords, pix_coords, found_obj = pomdp.update(
                vlm,
                molmo_tools,
                angle,
                camera_pos,
                cam_pose_normal,
                rgb,
                depth,
                RIG_config,
                tsdf_planner,
                cam_intr,
                cnt_step,
                pts,
                debug_f_path
            )

        # For debugging
        if found_obj:
            print("Pix Coords Shape: ", pix_coords)
            print("Real Coords Shape: ", real_world_coords)
            np.save(debug_f_path+f"img_{cnt_step}.npy", rgb)
            np.save(debug_f_path+f"pix_coords_{cnt_step}.npy", pix_coords)
            np.save(debug_f_path+f"real_coords_{cnt_step}.npy", real_world_coords)


        ########################
        # Determine next point #
        ########################
        print("Starting VLM check")
        pts_normal, angle, pts_pix, fig = get_next_point(
                rgb,
                question,
                pts_normal,
                cam_pose_tsdf,
                camera_pos,
                cam_pose_normal,
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
                RIG_config,
                pomdp
            )

        pts_pixs = np.vstack((pts_pixs, pts_pix))
        pts_normal = np.append(pts_normal, floor_height)
        pts = pts_normal

        ##############################
        # Check for early completion #
        ##############################
        done, symbolic_info = pomdp.enough_info(cnt_step)

        if done:
            break

        cnt_step += 1
        print("Done Iteration: ", cnt_step)

    # Get answer from POMDP
    # Check if success using weighted prediction
    _, symbolic_info = pomdp.enough_info(cnt_step)
    print("Done")
    print("Symbolic Info: ", symbolic_info)

    #Set up result and return
    result = get_result(
            prog,
            symbolic_info,
            question,
            pomdp,
            text_answer,
            cum_sim_score,
            cfg
        )

    result = {**result, **result_ans}

    return result
