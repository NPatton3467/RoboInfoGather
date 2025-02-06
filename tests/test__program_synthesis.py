import os
import csv

# RoboInfoGather Imports
from RoboInfoGather.program_utils import *

def main(cfg):
    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    
    # Run all questions
    for question_ind in range(10):
        # Extract question
        question_data = questions_data[question_ind]
        question = question_data["question"]
        
        # Generate program
        attempts = 0
        prog = None
        while attempts < 30:
            attempts += 1
            try:
                prog = gen_prog_from_nl(question)
                break
            except Exception as e:
                print(f"Could not generate program, attempt {attempts}")

        print(f"\nQuestion:\n{question}\nProgram: ")
        print(prog.pretty_str())

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    main(cfg)
