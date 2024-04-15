import numpy as np
import json

from map_utils import *

class ObstacleMap():
    def __init__(self, resolution, grid_range):
        self.resolution = resolution
        self.grid_range = grid_range
        self.obstacles = np.zeros((self.resolution, self.resolution))

    def update(self, obs, eps=1e-6):
        # Update Belief using Binary Bayes Filter
        p = self.p

        log_p = np.log(p/(1-p))

        inv_sensor_model = log((obs-eps)/(1-obs+eps))

        new_log_p = log_p + inv_sensor_model

        self.p = 1 - (1/(1+np.exp(new_log_p)))