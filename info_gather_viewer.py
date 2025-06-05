import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse
import imageio as iio

class StandaloneInfoGatherView:
    def __init__(self, floor_plan, env_img, gt_pts):
        self.floor_plan = floor_plan
        self.env_img = env_img
        self.gt_pts = gt_pts

        self.belief_maps = []         # list of (name, heatmap)
        self.plan = None              # list of (x, y)
        self.reachable_poses = None   # list of (x, y)
        self.trajectory = []          # list of (x, y)
        self.observations = []        # list of (x, y, label)

        self._build_gui()

    def add_belief_map(self, heatmap: np.ndarray, name: str):
        self.belief_maps.append((name, heatmap))

    def set_plan(self, path: list):
        self.plan = path

    def set_reachable_poses(self, poses: list):
        self.reachable_poses = poses

    def add_trajectory_point(self, x, y):
        self.trajectory.append((x, y))

    def add_observation(self, x, y, label):
        self.observations.append((x, y, label))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--scene")
    args = parser.parse_args()

    scene_name = args.scene
    floor_file = f"./og_scenes/scenes/{scene_name}/layout/floor_trav_0.png"
    env_file = f"./og_scenes/birds-eye-views/{scene_name}.png"

    floor = np.array(iio.imread(floor_file))

    # Extend to 3 Dims so that array sizes match
    floor = np.expand_dims(floor, axis=-1)
    floor = np.append(floor, np.zeros_like(floor), axis=-1)
    floor = np.append(floor, np.zeros_like(floor), axis=-1)

    # Make Alpha Max
    floor[:,:,3] = 255

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

    print("Floor shape: ", floor.shape)
    print("Env shape: ", env.shape)
    
    # Temp
    size = (100,100)
    gt_pts = [
        # Three around dining table
        (565, 390),
        (565, 510),
        (510, 450),

        # One by desk
        (420, 65)
    ]

    viewer = StandaloneInfoGatherView(floor, env, gt_pts)

    heatmap = np.random.rand(*size)
    viewer.add_belief_map(heatmap, "belief_a")
