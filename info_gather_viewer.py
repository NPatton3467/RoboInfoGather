import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse
import imageio as iio
import csv

class StandaloneInfoGatherView:
    def __init__(self, floor_plan, env_img, gt_pts, belief_maps):
        self.floor_plan = floor_plan
        self.env_img = env_img
        self.gt_pts = gt_pts

        self.belief_maps = belief_maps # list of (name, heatmap)
        self.plan = None               # list of (x, y)
        self.reachable_poses = None    # list of (x, y)
        self.trajectory = []           # list of (x, y)
        self.observations = []         # list of (x, y, label)

        self._build_gui()

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("InfoGatherView")

        # Canvas and figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2)

        # Layer checkboxes
        self.vars = {
            "floor_plan": tk.BooleanVar(value=True),
            "env": tk.BooleanVar(value=False),
            "gt": tk.BooleanVar(value=False),
            "belief": tk.BooleanVar(value=False),
            #"plan": tk.BooleanVar(value=True),
            #"reachable": tk.BooleanVar(value=True),
            #"trajectory": tk.BooleanVar(value=True),
            #"observations": tk.BooleanVar(value=True)
        }

        row = 1
        for i, (key, var) in enumerate(self.vars.items()):
            cb = tk.Checkbutton(self.root, text=key.replace("_", " ").title(), variable=var)
            cb.grid(row=row + i // 2, column=i % 2, sticky='w')

        # Button
        btn = tk.Button(self.root, text="Update View", command=self.draw)
        btn.grid(row=row + 4, column=0, columnspan=2)

        # Initial draw
        self.draw()
        self.root.mainloop()

    def draw(self):
        self.ax.clear()
        base = None

        if self.vars["floor_plan"].get():
            if base is None:
                base = self.floor_plan
            else:
                base = cv2.addWeighted(base, 0.5, self.floor_plan, 0.5, 0)

        if self.vars["env"].get():
            if base is None:
                base = self.env_img
            else:
                base = cv2.addWeighted(base, 0.5, self.env_img, 0.5, 0)

        self.ax.imshow(base)

        if self.vars["gt"].get():
            for pt in self.gt_pts:
                self.ax.scatter(pt[0], pt[1], marker="o", color="blue")

        if self.vars["belief"].get():
            for name, heatmap in self.belief_maps:
                sns.heatmap(heatmap, alpha=0.4, cmap='Reds', ax=self.ax, cbar=False)

        """
        if self.vars["reachable"].get() and self.reachable_poses:
            xs, ys = zip(*self.reachable_poses)
            self.ax.scatter(xs, ys, c='cyan', s=10, label='Reachable Poses')

        if self.vars["plan"].get() and self.plan:
            xs, ys = zip(*self.plan)
            self.ax.plot(xs, ys, '-g', linewidth=2, label='Plan')

        if self.vars["trajectory"].get() and self.trajectory:
            xs, ys = zip(*self.trajectory)
            self.ax.plot(xs, ys, '--b', linewidth=2, label='Trajectory')

        if self.vars["observations"].get():
            for x, y, label in self.observations:
                self.ax.text(x, y, label, color='yellow', fontsize=8)
        """

        self.ax.legend()
        self.ax.set_title("InfoGather Visualization")
        self.canvas.draw()

def load_q_data(question_data_path):
    # Load the dataset and get the task indexed by the arguments
    with open(question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]

    return questions_data

def get_floor(floor_file):
    floor = np.array(iio.imread(floor_file))

    # Extend to 3 Dims so that array sizes match
    floor = np.expand_dims(floor, axis=-1)
    floor = np.append(floor, np.zeros_like(floor), axis=-1)
    floor = np.append(floor, np.zeros_like(floor), axis=-1)

    # Make Alpha Max
    floor[:,:,3] = 255

    return floor

def get_env_bev(env_file):
    env = np.rot90(np.flip(np.array(iio.imread(env_file)), axis=1), k=3)

    # Delete 0 Alpha padding in env
    while (env[0,:,3] == 0).all():
        env = env[1:, :, :]

    while (env[-1,:,3] == 0).all():
        env = env[:-1, :, :]

    while (env[:, 0, 3] == 0).all():
        env = env[:, 1:, :]

    while (env[:, -1, 3] == 0).all():
        env = env[:, :-1, :]

    # Resize to be the same as the floor map
    env = cv2.resize(env, (int(env.shape[1] * (floor.shape[0]/env.shape[0])), floor.shape[0]))
    pad_width = floor.shape[1] - env.shape[1]
    env = np.pad(env, ((0,0), (0,pad_width), (0,0)))

    return env

def get_bel_maps(task_index, shape):
    # TODO: Expand to cover multiple beliefs for same task
    belief_file = f"./temp_debug/{task_index}/bel_Chair_71.npy" # TODO: Shouldn't be hard coded when using
                                                                # but naming convention will change
    chair_near_table_belief = np.mean(np.load(belief_file), axis=-1)

    chair_near_table_belief = cv2.resize(chair_near_table_belief, 
                                    (shape[1], shape[0])
                                )

    belief_maps = [
            ('Chair near Table', chair_near_table_belief)
        ]

    return belief_maps

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--task")
    parser.add_argument("-d", "--data_set")
    args = parser.parse_args()

    task_index = int(args.task)
    data_set = args.data_set

    # Load the data set
    question_data_path = f"./data/{data_set}_questions.csv"
    questions_data = load_q_data(question_data_path)
    task = questions_data[task_index]
    scene_name = task['scene']

    # Get the files to load floor and environment BEV
    floor_file = f"./og_scenes/scenes/{scene_name}/layout/floor_trav_0.png"
    env_file = f"./og_scenes/birds-eye-views/{scene_name}.png"

    # Get the floor
    floor = get_floor(floor_file)

    # Get the env BEV
    env = get_env_bev(env_file)

    print("Floor shape: ", floor.shape)
    print("Env shape: ", env.shape)
    
    # Temp ground truth points
    gt_pts = [
        # Three around dining table
        (565, 390),
        (565, 510),
        (510, 450),

        # One by desk
        (420, 65)
    ]
   
    # Get belief maps
    belief_maps = get_bel_maps(task_index, env.shape)

    viewer = StandaloneInfoGatherView(floor, env, gt_pts, belief_maps)
