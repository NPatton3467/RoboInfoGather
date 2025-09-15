"""
Run EQA in OmniGibson with RoboInfoGather exploration.

"""

# General Tool Imports
import os
import subprocess
import time

# RoboInfoGather Imports
from RoboInfoGather.run_RIG import *

def main(cfg, cfg_filename):
    # Load dataset
    questions_data = load_dataset(cfg)

    # Run all questions
    for question_ind in tqdm(range(len(questions_data))):
        if question_ind > 0:
            continue
        # Call bash script to run run_RIG.py with current question index
        print(f"./stream_script_new.sh python RoboInfoGather/run_RIG.py -cf {cfg_filename} -qind {str(question_ind)}")
        completed_process = subprocess.run(["./stream_script_new.sh",
                                            "python",
                                            "RoboInfoGather/run_RIG.py",
                                            "-cf", cfg_filename,
                                            "-qind", str(question_ind)])

        print(completed_process)
        print(completed_process.returncode)


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
    main(cfg, args.cfg_file)
