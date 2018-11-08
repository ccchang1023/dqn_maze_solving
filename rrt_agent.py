import numpy as np
import os, random, copy,math
from collections import deque
from heapq import heappush, heappop
from maze import generate_robot_map, TEST_MAZE, DEFAULT_MAZE, generate_block_map
import numpy as np
import matplotlib.pyplot as plt


class NODE(object):
    def __init__(self, pos):
        self.pos = pos
        self.parent = None

class RRT(object):
    def __init__(self, **rrt_params):
        self.maze = generate_robot_map()
        # self.maze = DEFAULT_MAZE
        # self.maze = generate_block_map(size=40)
        print(self.maze)
        nrows, ncols = np.shape(self.maze)
        self.reset()

    def reset(self):
        nrows, ncols = np.shape(self.maze)
        self.road_list = [[x, y] for x in range(nrows) for y in range(ncols) if self.maze[x, y] == 1]
        self.pos = [0,0]
        self.goal = [0,ncols-1]
        self.tree = []
        self.tree.append(NODE(self.pos))

    def gen_path(self, sample_num=1000, delta=0.3):
        x_size, y_size = np.shape(self.maze)
        goal_node = NODE(self.goal)
        answer = []

        #Initial exploring tree
        # while len(self.tree) < 10:
        #     idx_list =  np.random.randint(0,len(self.road_list))
        #     n = NODE(self.road_list[idx_list])
        #     self.connect_node(target_node=n, delta=delta)
            # self.show_exploring_tree()
        while len(self.tree) < sample_num:
            idx = np.random.randint(0,len(self.road_list))
            n = NODE(self.road_list[idx])
            self.connect_node(target_node=n, delta=delta)
            #Check if it can reach the goal
            node = self.get_nearest_node(goal_node)
            if len(self.get_direct_path(node.pos, self.goal)) != 0:
                while(node!= None):
                    answer.append(node.pos)
                    node = node.parent
                break
            # self.show_exploring_tree()
        return answer


    def connect_node(self, target_node, delta=1):
        # print("current tree:")
        # for p in self.tree:
        #     print(p.pos)
        # print("target node : ")
        # print(target_node.pos)
        nearest_node = self.get_nearest_node(target_node)
        # print("Nearest Node :", nearest_node.pos)
        xn,yn = nearest_node.pos
        xt,yt = target_node.pos
        new_pos = [math.ceil(xn+(xt-xn)*delta),math.ceil(yn+(yt-yn)*delta)]
        # print("new pos:", new_pos)

        if len(self.get_direct_path(nearest_node.pos, new_pos)) != 0:
            self.road_list.remove(new_pos)
            n = NODE(new_pos)
            n.parent = nearest_node
            self.tree.append(n)

    def get_nearest_node(self, target):
        min_idx = 0
        min_dist = 50000
        for i in range(len(self.tree)):
            p1 = self.tree[i].pos
            p2 = target.pos
            dist = math.sqrt(abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return self.tree[min_idx]


    def get_direct_path(self, start, target):
        ans = []
        s = start.copy()
        while s != target:
            if abs(s[0]-target[0]) >= abs(s[1]-target[1]):
                if target[0] > s[0]:
                    s[0] += 1
                else:
                    s[0] -= 1
            else:
                if target[1] > s[1]:
                    s[1] += 1
                else:
                    s[1] -= 1
            #Check if there's obstacle
            if self.maze[s[0]][s[1]] == 0:
                return []
            ans.append(s)
        return ans


    def show_exploring_tree(self):
        tmp_maze = self.maze.copy()
        for node in self.tree:
            x,y = node.pos
            tmp_maze[x][y] = -1
        print(tmp_maze)











