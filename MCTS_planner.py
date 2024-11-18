import math
import json
import time
import random

import numpy as np

from enum import Enum

from RoboInfoGather.map_utils import *

import cv2

class Loc():
    def __init__(self, x, y, theta=None):
        self.x = x
        self.y = y
        self.theta = theta

class Action(Enum):
    M_LEFT = 1
    M_RIGHT = 2
    M_DOWN = 3
    M_UP = 4
    M_LEFTDOWN = 5
    M_LEFTUP = 6
    M_RIGHTDOWN = 7
    M_RIGHTUP = 8
    L_N = 9
    L_NE = 10
    L_E = 11
    L_SE = 12
    L_S = 13
    L_SW = 14
    L_W = 15
    L_NW = 16
    OBS = 17

"""
class Action(Enum):
    M_FORWARD = 0
    M_BACKWARD = 1
    R_CCW = 2
    R_CW = 3
    OBS = 4
"""

class MCTS_Tree_Node():
    def __init__(self, loc, obstacle_map, num_prev_obs, max_obs, config,
            parent=None, children=[], inbound_act=None, terminal=False):

        
        self.loc = loc

        self.obstacle_map = obstacle_map

        self.num_prev_obs = num_prev_obs
        self.max_obs = max_obs

        self.config = config

        self.children = children
        self.parent = parent
        self.inbound_act = inbound_act
        self.terminal = terminal

        self.total_rewards = 0
        self.visits = 1

        self.max_children = len(Action)

        self.is_legal = None
        self.is_legal = self.legal(Action.OBS)

        # Find number of illegal actions and subtract from max children
        # Robot Must Start on Map Legally****
        if self.parent == None:
            print("End of MCTS Root INIT... Finding LEGAL BELOW:")
            print("Current number of Children: ", len(self.children))
            print("Input Number of Children: ", len(children))
        for act in Action:
            if not self.legal(act) and self.is_legal:
                self.max_children -= 1

    def print_node(self, depth=0):
        delim_str = ""

        if depth > 2:
            return

        for i in range(depth):
            delim_str += '| '

        print(delim_str, "Visits: ", self.visits, " Total Reward: ", self.total_rewards, " Inbound Act: ", self.inbound_act, " Number of children: ", len(self.children))
        for child in self.children:
            child.print_node(depth+1)

    def unvisited_child(self):
        avail_act_list = []
        for act in Action:
            found = False
            for child in self.children:
                if child.inbound_act == act:
                    found = True
                    break

            if not found and self.legal(act):
                avail_act_list.append(act)

        # Randomly select from available actions
        act = random.choice(avail_act_list)

        # Make new node
        new_loc = self.get_loc(act)
        
        # Increment number of previous observations if current node is an
        # Observe action
        new_num_prev_obs = self.num_prev_obs 
        if self.inbound_act == Action.OBS:
            new_num_prev_obs += 1

        child = MCTS_Tree_Node(
                loc = new_loc,
                obstacle_map = self.obstacle_map,
                num_prev_obs = new_num_prev_obs,
                max_obs = self.max_obs,
                config = self.config,
                parent = self,
                children = [],
                inbound_act = act,
                terminal = False)

        child.terminal = child.eval_terminal()

        # Add child to children and return
        self.children.append(child)
        if self.parent == None:
            print("Adding new child to root. Location: ", self.children[-1].loc.x, self.children[-1].loc.y, self.children[-1].loc.theta)
        return child

    def eval_terminal(self):
        # TODO THINK ABOUT THIS MORE
        #return (self.num_prev_obs == (self.max_obs - 1)) and self.inbound_act == Action.OBS
        terminal = ((self.inbound_act == Action.OBS) )# or
                    #self.inbound_act == Action.L_N) or
                    #self.inbound_act == Action.L_NE) or
                    #self.inbound_act == Action.L_E) or
                    #self.inbound_act == Action.L_SE) or
                    #self.inbound_act == Action.L_S) or
                    #self.inbound_act == Action.L_SW) or
                    #self.inbound_act == Action.L_W) or
                    #self.inbound_act == Action.L_NW))
        return terminal

    def legal(self, act):
        # Allow "illegal" actions if robot is stuck in inflated obstacle zone
        if self.is_legal != None and not self.is_legal:
            return True

        new_loc = self.get_loc(act)

        xy = [new_loc.x, new_loc.y]

        mxy = world_to_map(xy, self.obstacle_map.resolution, self.obstacle_map.size)

        dilate_rad = int(np.ceil(0.075 / self.obstacle_map.resolution))
        local_obs_map = cv2.dilate(self.obstacle_map.obstacles, np.ones((dilate_rad,dilate_rad)))

        # Check that new location is within the map bounds
        if mxy[0] < 0 or mxy[0] >= self.obstacle_map.size or\
            mxy[1] < 0 or mxy[1] >= self.obstacle_map.size:
            return False
        
        # Check if new location would cause a collision
        if local_obs_map[mxy[0], mxy[1]] > 0:
            return False

        return True

    def get_loc(self, act):
        j_size = self.config['planner_params']['mcts_step_length']
        if act == Action.M_LEFT:
            cur_map_loc = world_to_map([self.loc.x, self.loc.y], self.obstacle_map.resolution, self.obstacle_map.size)
            cur_map_loc[0] = int(cur_map_loc[0] - j_size/self.obstacle_map.resolution)

            new_real_loc_xy = map_to_world(cur_map_loc, self.obstacle_map.resolution, self.obstacle_map.size)

            return Loc(new_real_loc_xy[0], new_real_loc_xy[1], self.loc.theta)
        elif act == Action.M_RIGHT:
            cur_map_loc = world_to_map([self.loc.x, self.loc.y], self.obstacle_map.resolution, self.obstacle_map.size)
            cur_map_loc[0] = int(cur_map_loc[0] + j_size/self.obstacle_map.resolution)

            new_real_loc_xy = map_to_world(cur_map_loc, self.obstacle_map.resolution, self.obstacle_map.size)

            return Loc(new_real_loc_xy[0], new_real_loc_xy[1], self.loc.theta)
        elif act == Action.M_DOWN:
            cur_map_loc = world_to_map([self.loc.x, self.loc.y], self.obstacle_map.resolution, self.obstacle_map.size)
            cur_map_loc[1] = int(cur_map_loc[1] - j_size/self.obstacle_map.resolution)

            new_real_loc_xy = map_to_world(cur_map_loc, self.obstacle_map.resolution, self.obstacle_map.size)

            return Loc(new_real_loc_xy[0], new_real_loc_xy[1], self.loc.theta)
        elif act == Action.M_UP:
            cur_map_loc = world_to_map([self.loc.x, self.loc.y], self.obstacle_map.resolution, self.obstacle_map.size)
            cur_map_loc[1] = int(cur_map_loc[1] + j_size/self.obstacle_map.resolution)

            new_real_loc_xy = map_to_world(cur_map_loc, self.obstacle_map.resolution, self.obstacle_map.size)

            return Loc(new_real_loc_xy[0], new_real_loc_xy[1], self.loc.theta)
        elif act == Action.M_LEFTDOWN:
            cur_map_loc = world_to_map([self.loc.x, self.loc.y], self.obstacle_map.resolution, self.obstacle_map.size)
            cur_map_loc[0] = int(cur_map_loc[0] - j_size/self.obstacle_map.resolution)
            cur_map_loc[1] = int(cur_map_loc[1] - j_size/self.obstacle_map.resolution)

            new_real_loc_xy = map_to_world(cur_map_loc, self.obstacle_map.resolution, self.obstacle_map.size)

            return Loc(new_real_loc_xy[0], new_real_loc_xy[1], self.loc.theta)
        elif act == Action.M_LEFTUP:
            cur_map_loc = world_to_map([self.loc.x, self.loc.y], self.obstacle_map.resolution, self.obstacle_map.size)
            cur_map_loc[0] = int(cur_map_loc[0] - j_size/self.obstacle_map.resolution)
            cur_map_loc[1] = int(cur_map_loc[1] + j_size/self.obstacle_map.resolution)

            new_real_loc_xy = map_to_world(cur_map_loc, self.obstacle_map.resolution, self.obstacle_map.size)

            return Loc(new_real_loc_xy[0], new_real_loc_xy[1], self.loc.theta)
        elif act == Action.M_RIGHTDOWN:
            cur_map_loc = world_to_map([self.loc.x, self.loc.y], self.obstacle_map.resolution, self.obstacle_map.size)
            cur_map_loc[0] = int(cur_map_loc[0] + j_size/self.obstacle_map.resolution)
            cur_map_loc[1] = int(cur_map_loc[1] - j_size/self.obstacle_map.resolution)

            new_real_loc_xy = map_to_world(cur_map_loc, self.obstacle_map.resolution, self.obstacle_map.size)

            return Loc(new_real_loc_xy[0], new_real_loc_xy[1], self.loc.theta)
        elif act == Action.M_RIGHTUP:
            cur_map_loc = world_to_map([self.loc.x, self.loc.y], self.obstacle_map.resolution, self.obstacle_map.size)
            cur_map_loc[0] = int(cur_map_loc[0] + j_size/self.obstacle_map.resolution)
            cur_map_loc[1] = int(cur_map_loc[1] + j_size/self.obstacle_map.resolution)

            new_real_loc_xy = map_to_world(cur_map_loc, self.obstacle_map.resolution, self.obstacle_map.size)

            return Loc(new_real_loc_xy[0], new_real_loc_xy[1], self.loc.theta)
        elif act == Action.L_N:
            return Loc(self.loc.x, self.loc.y, math.radians(0))
        elif act == Action.L_NE:
            return Loc(self.loc.x, self.loc.y, math.radians(45))
        elif act == Action.L_E:
            return Loc(self.loc.x, self.loc.y, math.radians(90))
        elif act == Action.L_SE:
            return Loc(self.loc.x, self.loc.y, math.radians(135))
        elif act == Action.L_S:
            return Loc(self.loc.x, self.loc.y, math.radians(180))
        elif act == Action.L_SW:
            return Loc(self.loc.x, self.loc.y, math.radians(225))
        elif act == Action.L_W:
            return Loc(self.loc.x, self.loc.y, math.radians(270))
        elif act == Action.L_NW:
            return Loc(self.loc.x, self.loc.y, math.radians(315))
        elif act == Action.OBS:
            return Loc(self.loc.x, self.loc.y, self.loc.theta)


        """
        # Estimate location based on actions
        # Rotations are ~15 degrees
        # Motition is 0.5m

        step_deg = self.config['planner_params']['mcts_step_deg']
        step_len = self.config['planner_params']['mcts_step_length']

        # CCW Rotation
        if act == Action.R_CCW:
            new_loc = Loc(self.loc.x, self.loc.y, self.loc.theta + math.radians(step_deg))
            if self.parent == None:
                print("Action: ", act, " Node's Location: ", self.loc.x, self.loc.y, self.loc.theta)
                print("New Node's Location: ", new_loc.x, new_loc.y, new_loc.theta)
            return new_loc
        
        # CW Rotation
        if act == Action.R_CW:
            new_loc = Loc(self.loc.x, self.loc.y, self.loc.theta - math.radians(step_deg))
            if self.parent == None:
                print("Action: ", act, " Node's Location: ", self.loc.x, self.loc.y, self.loc.theta)
                print("New Node's Location: ", new_loc.x, new_loc.y, new_loc.theta)
            return new_loc

        # Move Forward
        if act == Action.M_FORWARD: 
            new_x = self.loc.x + np.cos(self.loc.theta) * step_len
            new_y = self.loc.y + np.sin(self.loc.theta) * step_len
            # NEEDS TO BE BASED ON ANGLE
            new_loc = Loc(new_x, new_y, self.loc.theta)
            
            if self.parent == None:
                print("Action: ", act, " Node's Location: ", self.loc.x, self.loc.y, self.loc.theta)
                print("New Node's Location: ", new_loc.x, new_loc.y, new_loc.theta)
            return new_loc

        # Move Backward
        if act == Action.M_BACKWARD:
            new_x = self.loc.x - np.cos(self.loc.theta) * step_len
            new_y = self.loc.y - np.sin(self.loc.theta) * step_len
            # NEEDS TO BE BASED ON ANGLE
            new_loc = Loc(new_x, new_y, self.loc.theta)
            
            if self.parent == None:
                print("Action: ", act, " Node's Location: ", self.loc.x, self.loc.y, self.loc.theta)
                print("New Node's Location: ", new_loc.x, new_loc.y, new_loc.theta)
            return new_loc

        #Observation
        if act == Action.OBS:
            new_loc = Loc(self.loc.x, self.loc.y, self.loc.theta)
            
            if self.parent == None:
                print("Action: ", act, " Node's Location: ", self.loc.x, self.loc.y, self.loc.theta)
                print("New Node's Location: ", new_loc.x, new_loc.y, new_loc.theta)
            return new_loc

        """



    def update(self, reward):
        self.total_rewards += reward
        self.visits += 1

class MCTS_Planner():
    def __init__(self, 
            start,
            pomdp,
            obstacle_map,
            config,
            epsilon=1e-1,
            rollout_policy="Random",
            max_rollout_depth=50):

        print("Initializing New Planner")

        root_loc = Loc(start.x, start.y, start.theta)
        print("Root Location", root_loc.x, root_loc.y, root_loc.theta)

        self.config = config
        self.max_obs = self.config['planner_params']['max_observations']
        self.root = MCTS_Tree_Node(root_loc,
                obstacle_map,
                0,
                self.max_obs,
                config,
                parent=None,
                children=[],
                inbound_act=None,
                terminal=False)

        self.pomdp = pomdp

        self.obstacle_map = obstacle_map

        self.max_time = self.config['planner_params']['max_time']
        #self.epsilon = epsilon
        self.epsilon = self.config['planner_params']['epsilon']

        self.rollout_policy_tp = rollout_policy
        self.max_rollout_depth = max_rollout_depth

    def search(self):
        start_time = time.time()

        print("Start of mcts seach, root location: ", self.root.loc.x, self.root.loc.y, self.root.loc.theta)

        while (time.time() - start_time) < self.max_time:
            leaf = self.traverse(self.root)
            sim_reward = self.rollout(leaf)
            self.backpropogate(leaf, sim_reward)

        print("Returning from Planner")
        print("Num Root Children: ", len(self.root.children))
        print("End of mcts seach, root location: ", self.root.loc.x, self.root.loc.y, self.root.loc.theta)
        for child in self.root.children:
            print("Child Act: ", child.inbound_act, " Reward: ", child.total_rewards, " Visits: ", child.visits)
            print("Child Location: ", child.loc.x, child.loc.y, child.loc.theta)

        # Print Tree
        self.root.print_node()

        return self.best_child(self.root)

    def traverse(self, node):
        while len(node.children) == node.max_children and not node.terminal:
            node = self.best_ucb(node)

        if node.terminal:
            return node
        else:
            return node.unvisited_child()

    def rollout(self, node):
        terminal = node.terminal
        depth = 0
        reward = 0
        # Calculate reward for all object types
        for obj_tp in self.pomdp.reward_funcs.keys():
            reward += self.pomdp.reward_funcs[obj_tp].eval(self.pomdp.bel[obj_tp], self.obstacle_map, self.root, node)
        
        while not terminal and depth < self.max_rollout_depth:
            node = self.rollout_policy(node)
            terminal = node.terminal
            
            # Calculate reward for all object types
            for obj_tp in self.pomdp.reward_funcs.keys():
                #reward += (0.99 ** depth) * self.pomdp.reward_funcs[obj_tp].eval(self.pomdp.bel[obj_tp], self.obstacle_map, self.root, node)
                reward += self.pomdp.reward_funcs[obj_tp].eval(self.pomdp.bel[obj_tp], self.obstacle_map, self.root, node)
            
            depth += 1


        return reward

    def rollout_policy(self, node):
        if self.rollout_policy_tp == "Random":
            # Select random action until legal discovered
            legal = False
            while not legal:
                act = random.choice(list(Action))
                legal = node.legal(act)

            # Make new child
            new_loc = node.get_loc(act)

            # Increment number of previous observations if current node is an
            # Observe action
            new_num_prev_obs = node.num_prev_obs 
            if node.inbound_act == Action.OBS:
                new_num_prev_obs += 1

            child = MCTS_Tree_Node(
                    loc = new_loc,
                    obstacle_map = node.obstacle_map,
                    num_prev_obs = new_num_prev_obs,
                    max_obs = node.max_obs,
                    config = self.config,
                    parent = node,
                    children = [],
                    inbound_act = act,
                    terminal = False)

            child.terminal = child.eval_terminal()

            # Append child to current node and return
            node.children.append(child)
            return child
        else:
            assert False # No others implemented
            return None

    def backpropogate(self, node, reward):
        # Return at root
        if node.parent == None:
            node.update(reward)
            return
        
        node.update(reward)

        self.backpropogate(node.parent, reward)

    def best_ucb(self, node):
        best_ucb_val = 0
        best_child = node.children[0]
        for child in node.children:
            ucb_val = self.calculate_ucb(child)
            if ucb_val > best_ucb_val:
                best_ucb_val = ucb_val
                best_child = child

        return best_child

    def calculate_ucb(self, node):
        mean = float(node.total_rewards)/node.visits
        explore_bonus = self.epsilon * math.sqrt(math.log(node.parent.visits)/node.visits)

        return mean + explore_bonus

    def best_child(self, node):
        #most_visits = 0
        #best_child = node.children[0]
        #for child in node.children:
        #    if child.visits > most_visits and child.inbound_act != Action.OBS:
        #        most_visits = child.visits
        #        best_child = child
        best_reward = 0
        best_child = node.children[0]
        for child in node.children:
            if (child.total_rewards / child.visits) > best_reward and child.inbound_act != Action.OBS:
                best_reward = child.total_rewards / child.visits
                best_child = child

        print("Best Child Location: ", best_child.loc.x, best_child.loc.y, best_child.loc.theta)

        return best_child
