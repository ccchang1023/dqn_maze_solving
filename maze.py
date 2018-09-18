from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
# from keras.optimizers import SGD , Adam, RMSprop
# from keras.layers import ReLU
# from keras.layers.advanced_activations import LeakyReLU, PReLU
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

from enum import Enum

DEFAULT_MAZE =  np.array([
    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
    [ 1.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 0.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  0.],
    [ 0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
    [ 0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.]
])

class DIR(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DONW = 3


class MAZE(object):
    def __init__(self, ):
        self.maze = DEFAULT_MAZE
        self.is_terminate = False
        nrows, ncols = np.shape(self.maze)
        self.road_list =  [[x,y] for x in range(nrows) for y in range(ncols) if self.maze[x,y] == 0.]
        self.state = random.choice(self.road_list)
        # self.state = [0,0]
        self.goal = [nrows, ncols]
        self.step_count = 0
        self.move_penalty = 0.
        self.score = 0.
        self.visited_set = set()
        r,c = self.state
        self.visited_set.add((r,c))
        print(self.maze)
        # print(self.road_list[-1])
                      
    
    def reset(self, maze=DEFAULT_MAZE):
        self.maze = maze
        self.is_terminate = False
        nrows, ncols = np.shape(self.maze)
        self.road_list =  [[x,y] for x in range(nrows) for y in range(ncols) if self.maze[x,y] == 0.]
        self.state = random.choice(self.road_list)
        self.step_count = 0
    
    def check_valid(self,r,c):
        nrows, ncols = np.shape(self.maze)
        return not (r < 0 or r >= nrows or c < 0 or c >= ncols)
    
    def move(self, dir):
        is_valid = True
        is_terminate = False
        reward = 0.
        self.step_count += 1
        
        if dir == DIR.LEFT:
            self.state[1] -= 1
        elif dir == DIR.UP:
            self.state[0] -= 1
        elif dir == DIR.RIGHT:
            self.state[1] += 1
        else:
            self.state[0] += 1
            
        print("State after move:", self.state)    
            
        r,c = self.state
        if not self.check_valid(r,c):
            print("Invalid!")
            is_valid = False
            reward -= 0.8
            return (is_valid, is_terminate, reward)
        
        #If move to a block
        if self.maze[r,c] == 1.0:
            print("Block!")
            reward = -0.75
            return (is_valid, is_terminate, reward)
        
        self.visited_set.add((r,c))
        
        #If achieve the goal
        if self.state == self.goal:
            reward = 1.
            is_terminate = True

        elif (r,c) in self.visited_set:
            reward = -0.25
        
        if not is_terminate:
            reward -= 0.04
       
        return (is_valid, is_terminate, reward)
        
        
    def show(self):
        plt.grid('on')
        nrows, ncols = np.shape(self.maze)
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = np.copy(self.maze)
        for row,col in self.visited_set:
            canvas[row,col] = 0.6
        rat_row, rat_col  = self.state
        canvas[rat_row, rat_col] = 0.3   # rat cell
        canvas[nrows-1, ncols-1] = 0.9 # cheese cell
        img = plt.imshow(canvas, interpolation='none', cmap='gray')
        return img
        
        
        
        
        
        
        