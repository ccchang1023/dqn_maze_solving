import numpy as np
import os, random, copy,math
from collections import deque
from heapq import heappush, heappop
from maze import generate_robot_map, TEST_MAZE, DEFAULT_MAZE, generate_map
import numpy as np
import matplotlib.pyplot as plt


class NODE(object):
    def __init__(self, pos, move_count=0, pre_dir=None):
        self.pos = pos
        self.move_count = move_count
        self.pre_dir = pre_dir
        self.heuristic = 0
        self.parent = None
        self.successor = []

    def set_successor(self, sList):
        self.successor = sList

    def set_heuristic(self, h):
        self.heuristic = h

    def __lt__(self, other):
        return self.heuristic < other.heuristic


class SEARCH_AGENT(object):
    def __init__(self, **search_params):
        # print(search_params)
        # self.algorithm = search_params.get('algorithm', "Astar")
        # self.depth = search_params.get('depth', None)
        # self.maze = generate_robot_map()
        self.maze = generate_map(size=10, road_ratio=0.5)
        # self.maze = DEFAULT_MAZE
        # self.maze = TEST_MAZE
        # print(self.maze)
        self.nrows, self.ncols = np.shape(self.maze)
        # self.road_list = [[x, y] for x in range(self.nrows) for y in range(self.ncols) if self.maze[x, y] == 1]
        self.reset()

    def set_maze(self, maze):
        self.maze = maze
        self.nrows, self.ncols = np.shape(self.maze)
        self.visited_table = np.zeros([self.nrows, self.ncols], dtype=int)
        # self.road_list = [[x, y] for x in range(self.nrows) for y in range(self.ncols) if self.maze[x, y] == 1]

    def reset(self):
        self.visited_table = np.zeros([self.nrows, self.ncols], dtype=int)


    def search(self, start_pos=None, goal = None):

        root = NODE(start_pos)
        x,y = start_pos
        self.visited_table[x][y] = 1
        return self.Astar(root,goal)

        # if self.algorithm == "dfs":
        #     if self.dfs(root, self.depth) == True:
        #         self.create_img(self.answer_path)
        #     else:
        #         print("Can't not find the find")
        # elif self.algorithm == 'bfs':
        #     self.bfs(root)
        # elif self.algorithm == 'Astar':
        #     return self.Astar(root)
        # return

    def get_successor(self, node, move_count=0):
        successor = []
        x,y = node.pos
        next_state = [[x,y-1],[x-1,y],[x,y+1],[x+1,y]]
        for i in range(len(next_state)):
            xn,yn = next_state[i]
            if self.is_valid(xn,yn) and not self.is_block(xn,yn) and self.visited_table[xn,yn]==0:
                self.visited_table[xn, yn] = 1
                n = NODE(pos=[xn,yn], move_count=move_count+1, pre_dir=i)
                successor.append(n)
        return np.array(successor)


    def dfs(self, node, depth):
        if node.pos == self.goal:
            self.answer_path = copy.deepcopy(self.path)
            print("Find!")
            print(self.answer_path)
            return True

        elif depth==0:
            return False

        node.set_successor(self.get_successor(node, self.visited_table))

        # print("Current node : ", node.pos)
        # print("Successor : ")
        # for n in node.successor:
        #     print(n.pos)
        # print("Vistied table:")
        # print(self.visited_table)
        # input("continue : ")

        for n in node.successor:
            self.path.append(n.pos)
            x, y = n.pos
            self.visited_table[x, y] = 1
            if self.dfs(n, depth-1) == True:
                return True
            self.path.pop()
            self.visited_table[x, y] = 0
        return False

    def bfs(self, node):
        q = deque()
        q.append(node)
        farest_dist = 0

        while len(q) != 0:
            node = q.popleft()
            if node.move_count > farest_dist:
                farest_dist = node.move_count
                print("Farest dist:", farest_dist)

            if node.pos == self.goal:
                print("Find goal!")
                ans = []
                while node.parent!=None:
                    ans.append(node.pos)
                    node = node.parent
                ans.reverse()
                print(ans)
                return True
            x,y = node.pos
            self.visited_table[x,y] = 1
            successor = self.get_successor(node, self.visited_table, node.move_count)
            for n in successor:
                n.parent = node
                q.append(n)

        return False

    def Astar(self, node, goal):
        heap = []
        heuristic = self.euclidean_distance(node.pos, goal)
        node.set_heuristic(heuristic)
        heappush(heap,node)
        # farest_dist = 0
        while len(heap) != 0:
            node = heappop(heap)

            # print("node pos:", node.pos)
            # print("node heuristic:", node.heuristic)
            # print("-------------------")

            # if node.move_count > farest_dist:
            #     farest_dist = node.move_count
                # print("Farest dist:", farest_dist)

            if node.pos == goal:
                # print("Find goal!")
                pos_list = []
                dir_list = []
                while node.parent!=None:
                    pos_list.append(node.pos)
                    dir_list.append(node.pre_dir)
                    node = node.parent
                pos_list.reverse()
                dir_list.reverse()
                return pos_list, dir_list

            # x,y = node.pos
            # self.visited_table[x,y] = 1
            successor = self.get_successor(node, node.move_count)
            for n in successor:
                n.parent = node
                heuristic = n.move_count + self.euclidean_distance(n.pos, goal)
                n.set_heuristic(heuristic)
                heappush(heap,n)

        return None,None


    def euclidean_distance(self,p1,p2):
        return math.sqrt(abs(p1[0]-p2[0])**2 + abs(p1[1]-p2[1])**2)


    def is_block(self, x,y):
        return True if self.maze[x,y] == 0 else False

    def is_valid(self,x,y):
        r,c = np.shape(self.maze)
        return True if x>=0 and x<r and y >=0 and y<c else False


    def create_img(self, answer):
        nrows, ncols = np.shape(self.maze)
        canvas = np.copy(self.maze).astype(float)
        visited_point = answer
        for x,y in visited_point:
            canvas[x,y] = 0.6
        x,y = self.goal
        canvas[x,y] = 0.9 # goal cell
        img = plt.imshow(canvas, interpolation='None', cmap='gray', vmin=0, vmax=1, animated=True)
        plt.show()


    # def show_animate(self):
    #     ani = animation.ArtistAnimation(self.fig, self.img_list, interval=50, blit=True, repeat_delay=1000)
    #     plt.show()
        # ani.save("test.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
        # ani.save("loop.mp4")


















