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
    [ 1,  1,  1,  1,  1,  1,  1,  0,  1,  1]
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
    def __init__(self, maze=DEFAULT_MAZE, num_of_actions=4, lower_bound=None):
        self.maze = maze
        self.num_of_actions = num_of_actions
        self.reward_lower_bound = lower_bound
        self.reset()
                      
    def reset(self, fix_goal=True):
        self.terminate_tag = False
        nrows, ncols = np.shape(self.maze)
        self.road_list =  [[x,y] for x in range(nrows) for y in range(ncols) if self.maze[x,y] == 1]
        self.token_pos = random.choice(self.road_list)
        self.goal = [-1,-1]
        if fix_goal:
            self.goal = [nrows-1, ncols-1]
        else:
            while self.goal != self.token_pos:
                self.goal = random.choice(self.road_list)
        self.move_count = 0
        self.optimal_move_count = DEFAULT_MAZE_ANSWER[self.token_pos[0],self.token_pos[1]]
        self.reward_sum = 0.
        self.visited_list = np.zeros(np.shape(self.maze))
        self.visited_list[self.token_pos[0], self.token_pos[1]] = 1
        # self.visited_set = set()
        # self.visited_set.add(tuple(self.token_pos))
        self.img_list = []
    
    def move(self, dir):
        goal_tag = False
        terminate_tag = False
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
            reward = -0.8
            self.token_pos = pos_before_move

        elif self.is_goal():
            reward = 1.
            goal_tag = terminate_tag = True

        elif self.is_visited(self.token_pos[0],self.token_pos[1]):
            # print("is_visited!")
            reward = -0.25

        else:
            self.visited_list[self.token_pos[0],self.token_pos[1]] = 1
            # self.visited_set.add(tuple(self.token_pos))
            reward = -0.04

        self.reward_sum += reward
        
        # if self.reward_sum < self.reward_lower_bound:
        #     terminate_tag = True
        
        return (self.get_state(), reward, goal_tag, terminate_tag)
    
    #Return 1D array(nrows*ncols)  maze: 0=block, 1=road, 2=token, 3=goal, 4=visited
    def get_state(self):

        #state1: only maze
        # state = np.copy(self.maze)
        # for r,c in self.visited_set:
        #     state[r][c] = 4
        # r,c = self.token_pos
        # state[r][c] = 2
        # return state.reshape(1,-1) #In order to match with Keras input, check model input_shape

        #state2: visited_list + token_pos
        # return (np.append(self.visited_list, self.token_pos)).reshape(1,-1)

        #state3: token_pos + goal + maze +visited_list
        state = np.append(self.token_pos, self.goal)
        # state = np.append(state, np.copy(self.maze))
        return (np.append(state, self.visited_list)).reshape(1,-1)
        
        #state4: maze + move_count + reward_sum
        # state = np.copy(self.maze)
        # for r,c in self.visited_set:
            # state[r][c] = 4
        # r,c = self.token_pos
        # state[r][c] = 2
        # state = np.append(state, self.move_count)
        # state = np.append(state, self.reward_sum).astype(float)
        # return state.reshape(1,-1)

        #State5: For conv2d, maze+valid, return shape=(1,row, col, 1)
        # state = np.copy(self.maze)
        # for r,c in self.visited_set:
        #     state[r][c] = 4
        # r,c = self.token_pos
        # state[r][c] = 2
        # return state.reshape(1,state.shape[0],state.shape[1],1)

    def get_num_of_actions(self):
        return self.num_of_actions

    def get_token_pos(self):
        return self.token_pos
        
    def get_move_count(self):
        return self.move_count
     
    def get_optimal_move_count(self):
        return self.optimal_move_count
     
    def get_reward_sum(self):
        return self.reward_sum
    
    def get_reward_lower_bound(self):
        return self.reward_lower_bound
    
    def get_optimal_solution_diff(self):
        move_count = self.get_move_count()
        optimal_move_count = self.get_optimal_move_count()
        diff = move_count - optimal_move_count
        return diff
        # print("Moves:%d Answer:%d Diff:%d" %(move_count, optimal_move_count, diff))
    
    def is_block(self):
        r, c = self.token_pos
        return self.maze[r,c] == 0
        
    def is_valid(self):
        r, c = self.token_pos
        nrows, ncols = np.shape(self.maze)
        return not (r < 0 or r >= nrows or c < 0 or c >= ncols)
        
    def is_goal(self):
        return self.token_pos == self.goal
    
    def is_visited(self, x, y):
        return (self.visited_list[x,y]==1)
        # return tuple(self.token_pos) in self.visited_set

    def show_maze(self):
        x,y = self.token_pos
        m = np.copy(self.maze)
        m[x][y] = 2
        print(m)

    
    def generate_map(self, size=10, road_ratio=0.7):
        m = np.zeros([size,size],dtype=int)
        
        #Initialize in down stair ways
        # for x in range(size):
            # for y in range(size):
                # if x==y:
                    # m[x][y] = 1
                    # y += 1
                # elif x==y-1:
                    # m[x][y] = 1
                    # x += 1        
        
        #Initialize in 'U' way
        x = y = 0
        while x < size:
            m[x][y] = 1
            x += 1
        x -= 1
        while y < size:
            m[x][y] = 1
            y += 1
        y -= 1
        while x >= 0:
            m[x][y] = 1
            x -= 1
        
        
        road_list = [[x,y] for x in range(size) for y in range(size) if m[x][y]==1.0]
        neighbor_list = np.array(self.get_block_neighbor_by_list(m, road_list))
        # print("neighbor_list shape:", np.shape(neighbor_list))
        # print(neighbor_list)
        x,y = random.choice(neighbor_list)
        nb = np.array(self.get_block_neighbor_by_point(m,x,y))
        # print("nb shape:", np.shape(nb))
        # print(nb)
        neighbor_list = np.append(neighbor_list,nb,axis=0)
        
        road_num = int(size*size*road_ratio)
        road_count = neighbor_list.shape[0]
        while road_count < road_num:
            randNum = random.randint(0,neighbor_list.shape[0]-1)
            x,y = neighbor_list[randNum]
            neighbor_list = np.delete(neighbor_list, randNum, axis=0)            
            m[x][y] = 1
            nb = np.array(self.get_block_neighbor_by_point(m,x,y))
            if nb.shape[0] != 0:
                neighbor_list = np.append(neighbor_list,nb,axis=0)
                road_count += 1
        # print("final shape:", neighbor_list.shape)
        # print(neighbor_list)
        print(m)

       
    def get_block_neighbor_by_list(self, maze, road_list):
        size = maze[0].size
        nb = list()
        for x,y in road_list:
            # print("size:", size)
            next = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
            for [r,c] in next:
                if r>=0 and r<size and c>=0 and c<size and maze[r][c] == 0 and not [r,c] in nb:
                    nb.append([r,c])
            # print(nb)
            # print(np.shape(nb))
        return nb

        
    def get_block_neighbor_by_point(self, maze, x, y):
        size = maze[0].size
        nb = list()
        next = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        for [r,c] in next:
            if r>=0 and r<size and c>=0 and c<size and maze[r][c] == 0 and not [r,c] in nb:
                nb.append([r,c])
        return nb

        
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
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
        