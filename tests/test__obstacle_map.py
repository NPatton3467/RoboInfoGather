from omnigibson.utils.transform_utils import quat2mat

from RoboInfoGather.pomdp_exec import *
from groundingdino.util.inference import load_model
from RoboInfoGather.program_utils import *
from RoboInfoGather.pomdp import *
from RoboInfoGather.obstacle_map import *

from matplotlib import pyplot as plt

 # Load the config
config_filename = os.path.join(f"./RoboInfoGather/info_gather.yaml")
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

# check if we want to quick load or full load the scene
load_options = {
    "Quick": "Only load the building assets (i.e.: the floors, walls, doors)",
    "Full": "Load all interactive objects in the scene",
}
load_mode = choose_from_options(options=load_options, name="load mode", random_selection=False)
if load_mode == "Quick":
    config["scene"]["load_object_categories"] = ["floors", "walls", "door"]

# Load the environment
env = og.Environment(configs=config)

# Allow user to move camera more easily
og.sim.enable_viewer_camera_teleoperation()

# Reset env before start? 
og.log.info("Resetting environment")
env.reset()

# Make default trav_map size
resolution = config['scene']['trav_map_resolution']
trav_map = get_trav_map(config['scene']['trav_map_path'], config['scene']['floor'], resolution, resolution)

plt.imshow(trav_map)
plt.show()

# Change lidar mounting
_, rob_ori = env.robots[0].get_position_orientation()
cur_scan_pos, _ =env.robots[0]._sensors['robot0:scan_link_Lidar_sensor'].get_position_orientation()
cur_scan_pos[2] += 0.1
env.robots[0]._sensors['robot0:scan_link_Lidar_sensor'].set_position_orientation(cur_scan_pos, rob_ori)

size, _ = trav_map.shape
obstacle_map = ObstacleMap(resolution, size)

action = OrderedDict([('robot0', [0, 1])])

# Run a simple loop and reset periodically
max_iterations = 1
for j in range(max_iterations):
    og.log.info("Resetting environment")
    env.reset()
    for i in range(1000):
        state, reward, done, info = env.step(action)
        # Update Obstacle map 
        lidar_sensor = env.robots[0]._sensors['robot0:scan_link_Lidar_sensor']
        scan = state['robot0']['robot0:scan_link_Lidar_sensor_scan']

        obstacle_map.update(lidar_sensor, scan)

        if i % 100 == 0:
            if action == OrderedDict([('robot0', [1, 0])]):
                action = OrderedDict([('robot0', [0, 1])])
            else:
                action = OrderedDict([('robot0', [1, 0])])
            obstacle_map.visualize()


# Always close the environment at the end
env.close()