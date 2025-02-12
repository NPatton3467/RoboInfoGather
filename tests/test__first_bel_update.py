"""
Run EQA in Habitat-Sim with RoboInfoGather exploration.

"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np

np.set_printoptions(precision=3)
import csv
import pickle
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner

# RoboInfoGather Imports
from groundingdino.util.inference import load_model
from RoboInfoGather.program_utils import *
from RoboInfoGather.pomdp import *
from RoboInfoGather.pomdp_exec import *
from RoboInfoGather.MCTS_planner import *
from RoboInfoGather.map_utils import *

def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    # Set up models
    CONFIG_PATH = os.path.join("/robodata/user_data/npatt/explore-eqa/RoboInfoGather/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    WEIGHTS_PATH = os.path.join("/robodata/user_data/npatt/explore-eqa/RoboInfoGather/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    dino_model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    dino_device = 'cuda'
    dino_model = dino_model.to(torch.device(dino_device))


    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    with open(cfg.init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }

    # Load VLM 
    vlm = VLM(cfg.vlm)
    
    # Run all questions
    cnt_data = 0
    cum_sim_score = 0
    for question_ind in tqdm(range(len(questions_data))):
    #for question_ind in range(1):
        
        # Extract question
        question_data = questions_data[question_ind]
        scene = question_data["scene"]
        floor = question_data["floor"]
        scene_floor = scene + "_" + floor
        question = question_data["question"]
        # This was first but doesn't work choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
        choices = question_data["choices"].split(",")
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
        print("Text Answer: ", text_answer)
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]

        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        scene_mesh_dir = os.path.join(
            cfg.scene_data_path, scene, scene[6:] + ".basis" + ".glb"
        )
        navmesh_file = os.path.join(
            cfg.scene_data_path, scene, scene[6:] + ".basis" + ".navmesh"
        )
        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": cfg.hfov,
        }
        sim_cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        pathfinder.load_nav_mesh(navmesh_file)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        pts = init_pts
        angle = init_angle

        # Floor - use pts height as floor height
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
        
        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
        )

        ################################
        # Set Up Planner from Question #
        ################################
        # TODO: Planner Setup
        
        # Load the config
        config_filename = os.path.join(f"/robodata/user_data/npatt/explore-eqa/RoboInfoGather/info_gather.yaml")
        RIG_config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        
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
        pos = pos_habitat_to_normal(pts)
        
        while True:
            attempts += 1
            try:
                prog = gen_prog_from_nl(question)
                if type(prog) is Prog:
                    pomdp = gen_pomdp_from_query(
                            query=prog.expressions[0],
                            pos=pos,
                            yaw=angle,
                            trav_map_og_size=size,
                            trav_map_og_res=resolution,
                            configs=RIG_config
                        )
                else:
                    pomdp = gen_pomdp_from_query(
                            query=prog,
                            pos=pos,
                            yaw=angle,
                            trav_map_og_size=size,
                            trav_map_og_res=resolution,
                            configs=RIG_config
                        )
                break
            except Exception as e:
                print(f"Could not generate program, attempt {attempts}")

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
        while cnt_step < 1:

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            agent_state.position = pts
            agent_state.rotation = rotation
            agent.set_state(agent_state)
            pts_normal = pos_habitat_to_normal(pts)


            # Update camera info
            sensor = agent.get_state().sensor_states["depth_sensor"]
            quaternion_0 = sensor.rotation
            translation_0 = sensor.position
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
            cam_pose[:3, 3] = translation_0
            cam_pose_normal = pose_habitat_to_normal(cam_pose)
            cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

            print("\n\nCAMERA STUFF:")
            camera_pos = pos_habitat_to_normal(translation_0)
            print("translation_0 (sensor.position): ", translation_0)
            print("cam_pose: ", cam_pose)
            print("cam_pos_normal: ", cam_pose_normal)
            print("cam_pose_tsdf: ", cam_pose_tsdf)
            print("camera_pos: ", camera_pos)
            print("pts: ", pts)
            print("pts_normal: ", pts_normal)
            print("\n\n")

            # Get observation at current pose - skip black image, meaning robot is outside the floor
            obs = simulator.get_sensor_observations()
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]

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
                vox_preds, _ = get_vox_preds(
                        vlm,
                        angle,
                        camera_pos,
                        cam_pose,
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
                print("Camera Pose: ", cam_pose)
                print("PTS: ", pts)

                # Do the same for each feature
                print("Starting Feature Update in run_RIG")
                for feature in pomdp.bel[obj_tp].feature_bels.keys():
                    # Get predictions for all voxels based on observations
                    print(pomdp.bel[obj_tp])
                    vox_preds, feature_ret_vals = get_vox_preds(
                            vlm,
                            angle,
                            camera_pos,
                            cam_pose,
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

            # END TODO: Belief Update

            ########################
            # Determine next point #
            ########################
            if cnt_step < num_step:
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
                    plt.close()

                # Visual prompting
                draw_letters = ["A", "B", "C", "D"]  # always four
                fnt = ImageFont.truetype(
                    "data/Open_Sans/static/OpenSans-Regular.ttf",
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

                prompt_lsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Please answer with a single letter."
                # logging.info(f"Prompt Exp: {prompt_text}")

                print("Done VLM prompt set up")

                actual_num_prompt_points = len(prompt_points_pix)
                if actual_num_prompt_points >= cfg.visual_prompt.min_num_prompt_points:
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
                        
                    elif num_black_in_a_row > 50:
                        cnt_step += 1
                        num_black_in_a_row += 1
                    else:
                        num_black_in_a_row += 1

                    # logging.info(f"Prompt Exp: {prompt_text}")
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
                        cur_depth = depth[py,px]
                        print("Getting Reward")
                        world_coords = get_world_coords_from_depth(px, py, cur_depth, camera_pos, angle, cam_intr)
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
                        lsv[prompt_point_ind] += 2*reward

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
                pts = pos_normal_to_habitat(pts_normal)

                rotation = quat_to_coeffs(
                    quat_from_angle_axis(angle, np.array([0, 1, 0]))
                    * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                ).tolist()

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
        query_exec_res = prog.execute(symbolic_info)
        print("Prog Result:\n", query_exec_res)

        # Get LLM to generate Natural Lanugage answer
        query_str = prog.pretty_str()
        nl_ans = get_nl_answer(query_exec_res, question, query_str)
        print("Natural Language Answer:\n", nl_ans)

        # Increment LLM similarity score
        print("Actual Answer: ", text_answer)
        raw_sim_score = eval_similarity(nl_ans, text_answer)
        # Extract Int Score
        try:
            sim_score = int(raw_sim_score)
        except Exception as e:
            print(f"Failed to make int from model output: {str(e)}")
            sim_score = 1

        cum_sim_score += (sim_score - 1)/4
        print("Current Cumulative Sim Score: ", cum_sim_score)
        print("Current Number of Questions: ", len(questions_data))
        print("Current LLM-Match %: ", cum_sim_score/len(questions_data)*100)




    print("Cumulative Sim Score: ", cum_sim_score)
    print("Number of Questions: ", len(questions_data))
    print("LLM-Match %: ", cum_sim_score/len(questions_data)*100)


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


    print(cfg)
    cfg.vlm.device = 'cuda:2'
    print(cfg)

    # run
    main(cfg)
