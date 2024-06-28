import numpy as np
import json

from matplotlib import pyplot as plt

from RoboInfoGather.map_utils import *
from RoboInfoGather.observation_utils import *

class ObstacleMap():
    def __init__(self, resolution, size):
        self.resolution = resolution
        self.size = size
        self.obstacles = np.zeros((self.size, self.size))

        self.cur_world_scan = None

    def update(self, scan_sensor, scan, eps=1e-6):
        # Get xy locations where obstacle's are detected
        obstacle_detection_locs = []
        i = 0
        # Grab vector of corresponding angles for each scan line
        angles = np.arange(
            -np.radians(scan_sensor.horizontal_fov / 2),
            np.radians(scan_sensor.horizontal_fov / 2),
            np.radians(scan_sensor.horizontal_resolution),
        )

        # Convert into 3D unit vectors for each angle
        unit_vector_laser = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angles])
        scan_laser = unit_vector_laser * (scan * (scan_sensor.max_range - scan_sensor.min_range) + scan_sensor.min_range)
        
        pos, ori = scan_sensor.get_position_orientation()
        scan_world = quat_to_rot(ori).dot(scan_laser.T).T + pos


        scan_world = np.squeeze(scan_world)
        
        
        self.cur_world_scan = []

        for xyz in scan_world:
            xy_obstacle_map = world_to_map((xyz[0], xyz[1]), self.resolution, self.size)

            if xy_obstacle_map[0] > 0 and xy_obstacle_map[0] < np.shape(self.obstacles)[0] and\
              xy_obstacle_map[1] > 0 and xy_obstacle_map[1] < np.shape(self.obstacles)[1]:

                if np.sqrt((xyz[0] - pos[0]) ** 2 + (xyz[1] - pos[1]) ** 2) < 0.5:
                    continue

                if xyz[2] < 0.02:
                    continue

                self.cur_world_scan.append(xyz)

                # Doing something wrong here. Lidar is mounted 90 degrees from flat (scaning into z axis)
                # My transforms ought to work here but they don't
                # I will need to remove the ground plane and robot LIDAR self collisions from this as well            
                p = self.obstacles[xy_obstacle_map[0], xy_obstacle_map[1]]
                log_p = np.log((p+eps)/(1-p))

                inv_sensor_model = np.log((1-eps)/(eps))

                new_log_p = log_p + inv_sensor_model

                self.obstacles[xy_obstacle_map[0], xy_obstacle_map[1]] = 1 - (1/(1+np.exp(new_log_p)))


    def visualize(self):
        plt.imshow(self.obstacles)
        plt.savefig('cur_obstacle_map.png')
