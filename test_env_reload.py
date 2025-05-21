"""
Run EQA in OmniGibson with RoboInfoGather exploration.

"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
import numpy as np

np.set_printoptions(precision=3)
import csv
import pickle
import logging
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import importlib
VLM = importlib.import_module('explore-eqa.src.vlm').VLM
TSDFPlanner = importlib.import_module('explore-eqa.src.tsdf').TSDFPlanner

# RoboInfoGather Imports
from RoboInfoGather.program_utils import *
from RoboInfoGather.pomdp import *
from RoboInfoGather.pomdp_exec import *
from RoboInfoGather.MCTS_planner import *
from RoboInfoGather.map_utils import *

import omnigibson as og
import omnigibson.lazy as lazy

from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options

# Don't use GPU dynamics and use flatcache for performance boost
from omnigibson.macros import gm
gm.HEADLESS = True
gm.REMOTE_STREAMING="webrtc"
gm.OMNIGIBSON_REMOTE_STREAMING="webrtc"
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from src.geom import get_cam_intr

def main(cfg):
    # Load the config
    config_filename = cfg.rig_config_file_path
    RIG_config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)


    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    #with open(cfg.init_pose_data_path) as f:
    #    init_pose_data = {}
    #    for row in csv.DictReader(f, skipinitialspace=True):
    #        init_pose_data[row["scene_floor"]] = {
    #            "init_pts": [
    #                float(row["init_x"]),
    #                float(row["init_y"]),
    #                float(row["init_z"]),
    #            ],
    #            "init_angle": float(row["init_angle"]),
    #        }
    #logging.info(f"Loaded {len(questions_data)} questions.")

    # Load VLM 
    vlm = VLM(cfg.vlm)
    vlm.model._supports_cache_class = False

    # Load Molmo
    # Set up MOLMO
    processor = AutoProcessor.from_pretrained(
        #'allenai/Molmo-72B-0924',
        #'allenai/MolmoE-1B-0924',
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    molmo_model = AutoModelForCausalLM.from_pretrained(
        #'allenai/Molmo-72B-0924',
        #'allenai/MolmoE-1B-0924',
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    molmo_tools = {'model': molmo_model, 'processor': processor}

    # Where Dino Model Was loaded -- set to none now because it's easier than 
    # refactoring
    dino_model = None
    
    # Run all questions
    cnt_data = 0
    results_all = []
    cum_sim_score = 0
    for question_ind in tqdm(range(len(questions_data))):
    #for question_ind in range(59, len(questions_data)):
    #for question_ind in range(1):
        plt.close('all')

        # Load the environment
        print("Question?\n", questions_data[question_ind])
        scene_name = questions_data[question_ind]['scene']
        scene_config_file = cfg.scene_config_path + f"{scene_name}.yaml"
        scene_config = yaml.load(open(scene_config_file, "r"), Loader=yaml.FullLoader)
        env = og.Environment(configs=scene_config)

        resolution = scene_config['scene']['trav_map_resolution']
        trav_map = get_trav_map(scene_config['scene']['trav_map_path'], scene_config['scene']['floor'], resolution, resolution)

        # Allow user to move camera more easily
        og.sim.enable_viewer_camera_teleoperation()

        # Reset env before start? 
        og.log.info("Resetting environment")
        env.reset()


        # Change lidar mounting
        _, rob_ori = env.robots[0].get_position_orientation()
        print(env.robots[0]._sensors.keys())
        cur_scan_pos, _ =env.robots[0]._sensors['rob:scan_link:Lidar:0'].get_position_orientation()
        cur_scan_pos[2] += 0.1
        env.robots[0]._sensors['rob:scan_link:Lidar:0'].set_position_orientation(cur_scan_pos, rob_ori)

        # Change Camera Mounting
        camera_pos, camera_ori = env.robots[0]._sensors['rob:eyes:Camera:0'].get_position_orientation()
        camera_pos[2] += 0.2
        env.robots[0]._sensors['rob:eyes:Camera:0'].set_position_orientation(camera_pos, camera_ori)
        camera_tilt = cfg.camera_tilt_deg * np.pi / 180
        # Explore EQA Ver -- img_height = cfg.img_height
        # Explore EQA Ver -- img_width = cfg.img_width
        img_width, img_height = env.robots[0]._sensors['rob:eyes:Camera:0'].camera_parameters["renderProductResolution"]
        #cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)
        cam_intr = np.array(env.robots[0]._sensors['rob:eyes:Camera:0'].intrinsic_matrix.detach().cpu())
        #print("OmniGibson Camera Intrinsic Matrix:\n", cam_intr)

        #horizontal_aperture = env.robots[0]._sensors['rob:eyes:Camera:0'].camera_parameters["cameraAperture"][0]
        #focal_length = env.robots[0]._sensors['rob:eyes:Camera:0'].camera_parameters["cameraFocalLength"]
        #hfov = 2 * math.atan(horizontal_aperture / (2 * focal_length))
        #img_width, img_height = env.robots[0]._sensors['rob:eyes:Camera:0'].camera_parameters["renderProductResolution"]
        #assert False #should have hfov be in degrees not rads here
        #cam_intr2 = get_cam_intr(hfov, img_height, img_width)

        #print("H Aperture: ", horizontal_aperture)
        #print("F len: ", focal_length)
        #print("HFOV: ", hfov)
        #print("Img Width: ", img_width)
        #print("Img Height: ", img_height)
        #print("\n\n")

        #print("Explore-EQA Camera Intrinsic Matrix:\n", cam_intr2)
        #assert False

        # Debug Path
        debug_f_path = f"/robodata/user_data/npatt/explore-eqa/RoboInfoGather/debug/{question_ind}/"
        debug_f_path = cfg.debug_path + f"{question_ind}/"
        if not os.path.isdir(debug_f_path):
            os.mkdir(debug_f_path)

        # Extract question
        question_data = questions_data[question_ind]
        scene = question_data["scene"]
        floor = question_data["floor"]
        scene_floor = scene + "_" + floor
        question = question_data["question"]
        # This was first but doesn't work choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
        choices = question_data["choices"].split("',")
        answer = question_data["answer"]
        text_answer = choices[0]
        if answer == "B":
            text_answer = choices[1]
        elif answer == "C":
            text_answer = choices[2]
        elif answer == "D":
            text_answer = choices[3]
        print("Choices: ", choices)
        print("Answer: ", answer)

        text_answer = text_answer.lstrip("'\\([ ")
        text_answer = text_answer.rstrip("'\\)] ")
        
        print("Text Answer: ", text_answer)
        init_pts, _ = env.robots[0].get_position_orientation()
        init_pts = np.array(init_pts.cpu().detach())
        pitch, roll, init_angle = env.robots[0].get_rpy()
        init_angle = init_angle.cpu().detach()
        logging.info(f"\n========\nIndex: {question_ind} Scene: {scene} Floor: {floor}")

        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(cfg.output_dir, str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)
        result = {"question_ind": question_ind}

        pts = init_pts
        angle = init_angle

        # Floor - use pts height as floor height
        floor_height = 0.01
        map_size = trav_map.shape[0]
        tsdf_bnds = np.array(
                [
                    [(-map_size/2.0)*resolution, (map_size/2.0)*resolution],
                    [(-map_size/2.0)*resolution, (map_size/2.0)*resolution],
                    [floor_height - 0.2, floor_height + 3.5]
                ])
        scene_size = (map_size * resolution) ** 2
        num_step = 1
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )
        
        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pts,
            init_clearance=cfg.init_clearance * 2,
        )

        ################################
        # Set Up Planner from Question #
        ################################
        # TODO: Planner Setup
         
        # Generate program
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
        pos = pts
        prog = None    
        while attempts < 10000:
            attempts += 1
            try:
                prog = gen_prog_from_nl(question)
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

        if attempts == 10000:
            continue

        print("\n\nProgram: ")
        print(prog.pretty_str())
        
        
        # END TODO: Planner Setup
        print("\n\nQuery")
        print(pomdp.query.pretty_str())

        # Run steps
        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        cnt_step = 0
        num_black_in_a_row = 0
        responses = []
        while cnt_step < num_step:
            logging.info(f"\n== step: {cnt_step}")

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")
            ori_to_send = Rotation.from_euler('xyz', [pitch, roll, angle], degrees=False).as_quat()
            env.robots[0].set_position_orientation(pts, ori_to_send)
            pts_normal = pts
            result[step_name] = {"pts": pts, "angle": angle}


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
            print("cam_pos_normal: ", cam_pose_normal)
            print("cam_pose_tsdf: ", cam_pose_tsdf)
            print("camera_pos: ", camera_pos)
            print("pts: ", pts)
            print("pts_normal: ", pts_normal)
            print("\n\n")

            # Get observation at current pose - skip black image, meaning robot is outside the floor
            action = OrderedDict([('rob', np.array([0 , 0]))])
            state, _, _, _, info = env.step(action) # Take Empty step to get observations
            rgb = np.array(state['rob']['rob:eyes:Camera:0']['rgb'].detach().cpu())
            depth = np.array(state['rob']['rob:eyes:Camera:0']['depth_linear'].detach().cpu())

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

            #################
            # Update Belief #
            #################
            # TODO: Belief Update
            for obj_tp in pomdp.bel.keys():
                print(pomdp.bel[obj_tp])
                # Get predictions for all voxels based on observations
                vox_preds, _, found_obj, pix_coords, real_world_coords = get_vox_preds(
                        vlm,
                        molmo_tools,
                        angle,
                        camera_pos,
                        cam_pose_normal,
                        pomdp.bel[obj_tp],
                        obj_tp,
                        rgb,
                        depth,
                        dino_model,
                        RIG_config,
                        tsdf_planner,
                        cam_intr,
                        iteration=cnt_step
                    )
                
                pomdp.bel[obj_tp].update(vox_preds)
                print("Camera Pose: ", cam_pose_normal)
                print("PTS: ", pts)

                # Do the same for each feature
                print("Starting Feature Update in run_RIG")
                for feature in pomdp.bel[obj_tp].feature_bels.keys():
                    # Get predictions for all voxels based on observations
                    print(pomdp.bel[obj_tp])
                    vox_preds, feature_ret_vals, found_obj, pix_coords, real_world_coords = get_vox_preds(
                            vlm,
                            molmo_tools,
                            angle,
                            camera_pos,
                            cam_pose_normal,
                            pomdp.bel[obj_tp],
                            obj_tp,
                            rgb,
                            depth,
                            dino_model,
                            RIG_config,
                            tsdf_planner,
                            cam_intr,
                            feature=feature,
                            iteration=cnt_step
                        )

                    pomdp.bel[obj_tp].update(vox_preds, feature=feature, feature_ret_vals=feature_ret_vals)

            print("Done Feature Update in run_RIG")

            # For debugging
            if found_obj:
                np.save(debug_f_path+f"img_{cnt_step}.npy", rgb)
                np.save(debug_f_path+f"pix_coords_{cnt_step}.npy", pix_coords)
                np.save(debug_f_path+f"real_coords_{cnt_step}.npy", real_world_coords)

            # END TODO: Belief Update

            ########################
            # Determine next point #
            ########################
            print("Starting VLM check")
            # Get VLM prediction
            rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
            prompt_question = (
                question
                + "\nAnswer with the option's letter from the given choices directly."
            )
            # logging.info(f"Prompt Pred: {prompt_question}")
            vlm_pred_candidates = ["A", "B", "C", "D"]
            
            # Get frontier candidates
            prompt_points_pix = []
            if cfg.use_active:
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

            # Visual prompting
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

            prompt_lsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Please answer with a single letter."
            # logging.info(f"Prompt Exp: {prompt_text}")

            print("Done VLM prompt set up")

            num_black_pixels = np.sum(
                    np.sum(rgb[:,:,0:3], axis=-1) == 0
            )  # sum over channel first

            if num_black_pixels < cfg.black_pixel_ratio * img_width * img_height:
                #if cnt_step > 1:
                #    print("Num Black: ", num_black_pixels)
                #    print("Shape: ", rgb.shape)
                #    print("Max: ", np.max(rgb))
                #    print("Mean: ", np.mean(rgb))

                #    nbp = np.sum(np.sum(rgb[:,:,0:3], axis=-1) == 0)

                #    print("Num Black w/o Alpha Channel: ", nbp)

                cnt_step += 1
                num_black_in_a_row = 0
                
                if cfg.save_obs:
                    plt.imsave(
                        os.path.join(episode_data_dir, "{}.png".format(cnt_step)), rgb
                    )
            elif num_black_in_a_row > 50:
                cnt_step += 1
                num_black_in_a_row += 1
                if cfg.save_obs:
                    plt.imsave(
                        os.path.join(episode_data_dir, "{}.png".format(cnt_step)), rgb
                    )
            else:
                num_black_in_a_row += 1
            
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

                    # I'm not sure why this wasn't needed in the Explore-EQA code?
                    if py < 0 or py >= depth.shape[0] or px < 0 or px >= depth.shape[1]:
                        continue

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
                    #TODO: lsv[prompt_point_ind] += 2*reward

                # Integrate semantics only if there is any prompted point
                tsdf_planner.integrate_sem(
                    sem_pix=lsv,
                    radius=1.0,
                    obs_weight=1.0,
                )  # voxel locations already saved in tsdf class

                print("Finishing semantic integration")

            print("Getting next pose")
            pts_normal, angle, pts_pix, fig = tsdf_planner.find_next_pose(
                pts=pts_normal,
                angle=angle,
                flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                **cfg.planner,
            )
            pts_pixs = np.vstack((pts_pixs, pts_pix))
            pts_normal = np.append(pts_normal, floor_height)
            pts = pts_normal

            # Show belief
            #b = pomdp.bel[next(iter(pomdp.bel.keys()))].p
            #b = np.mean(b, axis=2)
            #np.save(f'/robodata/user_data/npatt/explore-eqa/debug/updated_beliefs/{next(iter(pomdp.bel.keys()))}_{cnt_step}.npy', b)
            #np.save(f'/robodata/user_data/npatt/explore-eqa/debug/obstacle_maps/{cnt_step}.npy', tsdf_planner._tsdf_vol_cpu)

            ##############################
            # Check for early completion #
            ##############################
            done, symbolic_info = pomdp.enough_info(cnt_step)

            if done:
                break

        # TODO:
        # Get answer from POMDP
        # Check if success using weighted prediction
        _, symbolic_info = pomdp.enough_info(cnt_step)
        print("Done")
        print("Symbolic Info: ", symbolic_info)

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
        results_all.append(result)
        cnt_data += 1
        #if cnt_data % cfg.save_freq == 0:
        with open(
            os.path.join(cfg.output_dir, f"results_{cnt_data}.pkl"), "wb"
        ) as f:
            pickle.dump(results_all, f)

        print("About to clear og")
        og.clear()
        print("Done clearing og")


    # Save all data again
    with open(os.path.join(cfg.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results_all, f)
    with open(os.path.join(cfg.output_dir, "responses.txt"), "w") as f:
        f.write(str(responses))

    logging.info(f"\n== All Summary")
    logging.info(f"Number of data collected: {cnt_data}")
    
    print("Cumulative Sim Score: ", cum_sim_score)
    print("Current Number of Questions: ", question_ind+1)
    print("LLM-Match %: ", cum_sim_score/(question_ind+1)*100)


    with open(os.path.join(".", "pomdp.pkl"), "wb") as f:
        pickle.dump(pomdp, f)


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(cfg.output_dir, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)
