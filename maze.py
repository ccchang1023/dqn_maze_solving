from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
# from keras.optimizers import SGD , Adam, RMSprop
# from keras.layers import ReLU
# from keras.layers.advanced_activations import LeakyReLU, PReLU
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from enum import Enum

DEFAULT_MAZE = np.array([
    [ 1,  0,  1,  1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  0,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  0,  1,  1,  1,  1],
    [ 0,  0,  1,  0,  0,  1,  0,  1,  1,  1],
    [ 1,  1,  0,  1,  0,  1,  0,  0,  0,  1],
    [ 1,  1,  0,  1,  0,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  0,  0,  0,  0],
    [ 1,  0,  0,  0,  0,  0,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1,  1,  1,  0,  1,  3]
])

DEFAULT_MAZE_ANSWER = np.array([
    [ 40,  0,  36,  35,  34,  33, 32,  31,  30,  29],
    [ 39,  38, 37,  36,  35,  0,  31,  30,  29,  28],
    [ 40,  39, 38,  37,  36,  0,  30,  29,  28,  27],
    [ 0,   0,  39,  0,   0,   22,  0,  28,  27,  26],
    [ 16,  17,  0,  19,  0,   21,  0,  0,   0,   25],
    [ 15,  16,  0,  18,  0,   20,  21, 22,  23,  24],
    [ 14,  15,  16, 17,  18,  19,  20, 21,  22,  23],
    [ 13,  14,  15, 16,  17,  18,  0,   0,   0,   0],
    [ 12,  0,   0,   0,   0,   0,  4,   3,   2,   1],
    [ 11,  10,  9,   8,   7,   6,  5,   0,   1,   0]
]) 



class DIR(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DONW = 3


class Maze(object):
    def __init__(self, ):
        self.reset()
                      
    def reset(self, maze=DEFAULT_MAZE):
        self.maze = maze
        self.terminate_tag = False
        nrows, ncols = np.shape(self.maze)
        self.road_list =  [[x,y] for x in range(nrows) for y in range(ncols) if self.maze[x,y] == 1]
        self.token_pos = random.choice(self.road_list)
        # self.token_pos = self.road_list[0]
        self.goal = [nrows-1, ncols-1]
        self.move_count = 0
        self.reward_lower_bound = -0.5*maze.size
        self.reward_sum = 0.
        # self.visited_set = set()
        # self.visited_set.add(tuple(self.token_pos))
        self.img_list = []
    
    def move(self, dir):
        goal_tag = False
        terminate_tag = False
        reward = 0.
        self.move_count += 1
        pos_before_move = list(self.token_pos)
        # print("before move", pos_before_move)
        
        if dir == DIR.LEFT:
            self.token_pos[1] -= 1
        elif dir == DIR.UP:
            self.token_pos[0] -= 1
        elif dir == DIR.RIGHT:
            self.token_pos[1] += 1
        else:
            self.token_pos[0] += 1
            
        if not self.is_valid():
            # print("Invalid!")
            terminate_tag = True
            reward = -0.8
            self.token_pos = pos_before_move    
        
        elif self.is_block():
            # print("Block!")
            terminate_tag = True
            reward = -0.75
            self.token_pos = pos_before_move
        
        elif self.is_goal():
            reward = 1.
            goal_tag = terminate_tag = True
        else:
            reward -= 0.04
            
        # if self.is_visited():
        #     reward -= 0.25
        # else:
        #     self.visited_set.add(tuple(self.token_pos))
        
        self.reward_sum += reward
        
        if self.reward_sum < self.reward_lower_bound:
            terminate_tag = True
        
        return (self.get_state(), reward, goal_tag, terminate_tag)
    
    #Return 1D array(nrows*ncols)  1=road, 0=block, 2=token
    def get_state(self):
        state = np.copy(self.maze)
        r,c = self.token_pos
        state[r][c] = 2
        return state.reshape(1,-1) #In order to match with Keras input, check model input_shape
    
    def get_token_pos(self):
        return self.token_pos
        
    def get_move_count(self):
        return self.move_count
     
    def get_reward_sum(self):
        return self.reward_sum
        
    def is_block(self):
        r, c = self.token_pos
        return self.maze[r,c] == 0
        
    def is_valid(self):
        r, c = self.token_pos
        nrows, ncols = np.shape(self.maze)
        return not (r < 0 or r >= nrows or c < 0 or c >= ncols)
        
    def is_goal(self):
        return self.token_pos == self.goal
    
    # def is_visited(self):
        # return tuple(self.token_pos) in self.visited_set
    
    def create_img(self):
        plt.grid(True)
        nrows, ncols = np.shape(self.maze)
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = np.copy(self.maze).astype(float)
        # for row,col in self.visited_set:
            # canvas[row,col] = 0.6
        rat_row, rat_col  = self.token_pos
        canvas[rat_row, rat_col] = 0.3   # token cell
        canvas[nrows-1, ncols-1] = 0.9 # goal cell
        img = plt.imshow(canvas, interpolation='None', cmap='gray', animated=False)
        self.img_list.append([img])
        
    def show_animate(self):
        fig = plt.figure()
        ani = animation.ArtistAnimation(fig,self.img_list, interval=50, blit=True, repeat_delay=1000)
        plt.show()
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
        