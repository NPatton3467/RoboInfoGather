import tkinter as tk
from tkinter import ttk
import numpy as np
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse
import imageio as iio
import csv

from RoboInfoGather.tsdf_visualization_utils import *
from RoboInfoGather.map_utils import *

class StandaloneInfoGatherView:
    def __init__(self, floor_plan, env_img, gt_pts, belief_maps, tsdf, result_pts, plan_pts):
        self.floor_plan = floor_plan
        self.env_img = env_img
        self.gt_pts = gt_pts

        self.belief_maps = belief_maps # list of (name, heatmap)
        self.tsdf = tsdf
        self.result_pts = result_pts
        self.plan = plan_pts               # list of (x, y)
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
            "tsdf": tk.BooleanVar(value=False),
            "plan": tk.BooleanVar(value=False),
            "result": tk.BooleanVar(value=False),
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
    
    def crop_base(self, base):
        temp_floor = np.copy(self.floor_plan)

        v_diff_0 = 0
        h_diff_0 = 0
        v_diff_1 = 0
        h_diff_1 = 0
        while (temp_floor[0,:,0] == 0).all():
            temp_floor = temp_floor[1:,:,:]
            base = base[1:,:,:]
            v_diff_0 += 1
        while (temp_floor[-1,:,0] == 0).all():
            temp_floor = temp_floor[:-1,:,:]
            base = base[:-1,:,:]
            v_diff_1 += 1
        while (temp_floor[:,0,0] == 0).all():
            temp_floor = temp_floor[:,1:,:]
            base = base[:,1:,:]
            h_diff_0 += 1
        while (temp_floor[:,-1,0] == 0).all():
            temp_floor = temp_floor[:,:-1,:]
            base = base[:,:-1,:]
            h_diff_1 += 1

        return base, (h_diff_0, h_diff_1), (v_diff_0, v_diff_1)
    
    def reshape(self, heatmap, h_diffs, v_diffs):
        shape = heatmap.shape
        heatmap = heatmap[v_diffs[0]:(shape[0]-v_diffs[1]), h_diffs[0]:(shape[1]-h_diffs[1])]
        return heatmap

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

        base, h_diffs, v_diffs = self.crop_base(base)

        self.ax.imshow(base)

        if self.vars["gt"].get():
            for pt in self.gt_pts:
                self.ax.scatter(pt[0] - h_diffs[0], pt[1] - v_diffs[0], marker="o", color="blue")

        if self.vars["result"].get():
            for pt in self.result_pts:
                self.ax.scatter(pt[0] - h_diffs[0], pt[1] - v_diffs[0], marker="*", color="blue")


        if self.vars["belief"].get():
            for name, heatmap in self.belief_maps:
                heatmap = self.reshape(heatmap, h_diffs, v_diffs)
                sns.heatmap(heatmap, alpha=0.4, cmap='Reds', ax=self.ax, cbar=False)

        if self.vars['tsdf'].get():
            tsdf = self.reshape(self.tsdf, h_diffs, v_diffs)
            sns.heatmap(tsdf, alpha=0.4, cmap='Blues', ax=self.ax, cbar=True)

        if self.vars['plan'].get():
            for (x, y, dx, dy) in self.plan:
                self.ax.scatter(x - h_diffs[0], y - v_diffs[0], marker="*", color="green")
                self.ax.arrow(x-h_diffs[0],y-v_diffs[0],20*dx,20*dy)

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

def get_env_bev(env_file, make_new_bev=False, shape=None):
    if make_new_bev:
        # Magic numbers based on actual width of floor plan in trav map
        # need to investigate way to make this on the fly
        env = np.rot90(np.flip(np.array(iio.imread(env_file)), axis=1), k=3)
        env = cv2.resize(env, (597-18, 757-10))
        #env = cv2.resize(env, (5862-1314, 4686-2422))
        #env = cv2.resize(env, (7082-1849, 8600-2752))
        
        pad_width = shape[1] - env.shape[1] - 18
        pad_height = shape[0] - env.shape[0] - 10
        env = np.pad(env, ((10,pad_height), (18,pad_width), (0,0)))

        print(env.shape)
        print(shape)

        plt.imshow(env)
        plt.show()

        np.save('./new.npy', env)
    else:
        env = np.load(env_file)

    return env

def get_bel_maps(bel, shape):
    # TODO: Expand to cover multiple beliefs for same task
    belief = np.rot90(np.flip(bel, axis=1), k=1)
    #belief = np.mean(belief, axis=-1)
    belief = np.max(belief, axis=-1)

    belief = cv2.resize(belief, (shape[1], shape[0]))

    belief_maps = [
            ('bel', belief)
        ]

    return belief_maps

def get_tsdf(tsdf_file, shape=None):
    #tsdf = np.load(tsdf_file)
    tsdf = np.rot90(np.flip(np.load(tsdf_file), axis=1), k=1)
    #tsdf = find_zero_crossings(tsdf, axis=2)
    tsdf = tsdf[:,:,3]
    #tsdf = np.rot90(np.flip(tsdf, axis=1), k=3)
    #tsdf = cv2.resize(tsdf, (597-18, 757-10))
    
    #pad_width = shape[1] - tsdf.shape[1] - 18
    #pad_height = shape[0] - tsdf.shape[0] - 10
    #tsdf = np.pad(tsdf, ((10,pad_height), (18,pad_width)))
    tsdf = np.pad(tsdf, ((1,1), (1,1)))
    tsdf = cv2.resize(tsdf, (shape[0], shape[1]))

    print(tsdf.shape)
    print(shape)

    plt.imshow(tsdf)
    plt.show()

    return tsdf

def get_result_pts(pomdp, shape):

    # Execute query to get symbolic result
    # TEMP
    symbolic_info = pomdp.make_symbolic()
    print("Symbolic Info:\n", symbolic_info)
    symbolic_result = pomdp.query.execute(symbolic_info)
    print("Query Result:\n", symbolic_result)
    #symbolic_result = {'Chair': pomdp.make_symbolic()['Chair']}
    # END TEMP

    # Extract the locations of points in the environment frame
    result_world_locs = []
    for obj in symbolic_result:
        for inst in symbolic_result[obj]:
            result_world_locs.append((obj, symbolic_result[obj][inst]['location']))

    print(result_world_locs)

    # Get points in bel frame
    result_bel_pts = []
    for (obj, pt) in result_world_locs:
        vol_origin = pomdp.bel[obj].map_params['vol_origin']
        map_resolution = pomdp.bel[obj].map_params['res']
        z_resolution = pomdp.bel[obj].map_params['z_res']
        map_dim = pomdp.bel[obj].map_params['dim']
        bel_shape = pomdp.bel[obj].p.shape

        result_bel_pts.append(world_to_map(pt, vol_origin, map_resolution, z_resolution, map_dim) * shape / bel_shape)

    print(result_bel_pts)

    return result_bel_pts

def get_plan_pts(result_file, shape):
    with open(result_file, 'rb') as f:
        result = pkl.load(f)

    pomdp = result['pomdp']
   
    # Extract the locations of points in the environment frame
    result_world_locs = []
    for t_step in range(30):
        x = result[f'step_{t_step}']['pts'][0]
        y = result[f'step_{t_step}']['pts'][1]
        angle = result[f'step_{t_step}']['angle'].detach().cpu()
        print("ANGLE: ", angle)
        dx = np.cos(angle)
        dy = np.sin(angle)
        result_world_locs.append((x, y, dx, dy))

    # Get points in bel frame
    result_bel_pts = []
    for (x, y, dx, dy) in result_world_locs:
        obj = next(iter(pomdp.bel))
        vol_origin = pomdp.bel[obj].map_params['vol_origin']
        map_resolution = pomdp.bel[obj].map_params['res']
        z_resolution = pomdp.bel[obj].map_params['z_res']
        map_dim = pomdp.bel[obj].map_params['dim']
        bel_shape = pomdp.bel[obj].p.shape
        
        b_xyz = world_to_map(np.array([x,y,0]), vol_origin, map_resolution, z_resolution, map_dim) * shape / bel_shape
        #b_dxy = world_to_map(np.array([dx,dy,0]), vol_origin, map_resolution, z_resolution, map_dim) * shape / bel_shape

        #result_bel_pts.append((b_xyz[0], b_xyz[1], b_dxy[0], b_dxy[1]))
        result_bel_pts.append((b_xyz[0], b_xyz[1], dx, dy))

    print(result_bel_pts)

    return result_bel_pts

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--task")
    parser.add_argument("-d", "--data_set")
    args = parser.parse_args()

    task_index = int(args.task)
    data_set = args.data_set

    # Load the data set
    question_data_path = f"./RoboInfoGather/data/{data_set}_questions.csv"
    questions_data = load_q_data(question_data_path)
    task = questions_data[task_index]
    scene_name = task['scene']

    # Get the files to load floor and environment BEV
    #floor_file = f"./RoboInfoGather/og_scenes/scenes/{scene_name}/layout/floor_trav_0.png"
    floor_file = f"./RoboInfoGather/og_scenes/scenes/Rs_int/layout/floor_trav_0.png"
    #env_file = f"./RoboInfoGather/OmniGibsonBEV/{scene_name}_cubes_BEV.png"
    #env_file = f"./RoboInfoGather/OmniGibsonBEV/Rs_int_cubes_BEV.png"
    env_file = f"./RoboInfoGather/OmniGibsonBEV/{scene_name}.npy"

    # Get the floor
    floor = get_floor(floor_file)

    # Get the env BEV
    #env = get_env_bev(env_file, True, floor.shape)
    env = get_env_bev(env_file)

    print("Floor shape: ", floor.shape)
    print("Env shape: ", env.shape)
    
    # Get ground truth points
    #gt_pts_file = f'./RoboInfoGather/data/multiview_gt_pts/{task_index}.npy'
    #gt_pts = np.load(gt_pts_file)
    gt_pts = None

    exp_f_path = './RoboInfoGather/results/RIG_multi_view_perfect_perception_from_sim_exp'

    # Get result points
    result_file = f'{exp_f_path}/results_{task_index}.pkl'
    with open(result_file, 'rb') as f:
        result = pkl.load(f)

    pomdp = result['pomdp']
    result_pts = get_result_pts(pomdp, floor.shape)
   
    # Get belief maps
    task_objects = [
            'Plant',
            'Cabinet',
            'Cabinet',
            'Chair',
            'Cabinet',
            'Cabinet',
            'Chair',
            'Picture',
            'Light',
            'Light',
            'Light',
            'Couch',
            'Sink',
            'Sink',
            'Toilet',
            'Chair',
            'Lamp',
            'Chair',
            'Sink',
            'Bed',
            'Chair',
            'Table',
            'Shelf',
            'Toilet',
            'Table',
            'Sink',
            'Table',
            'Table',
            'Couch',
            'Couch',
            'Rug',
            'Shelf',
            'Shelf',
            'Rug',
            'Shelf'
        ]
   
    belief_maps = get_bel_maps(np.array(pomdp.bel[task_objects[task_index]].p.detach().cpu()), env.shape)

    # Get TSDF
    #tsdf_file = f'{exp_f_path}/debug/{task_index}/tsdf_volume_29.npy'
    #tsdf = get_tsdf(tsdf_file, shape=floor.shape)
    tsdf = None

    # Get viewpoints
    plan_pts = get_plan_pts(result_file, floor.shape)

    viewer = StandaloneInfoGatherView(floor, env, gt_pts, belief_maps, tsdf, result_pts, plan_pts)
