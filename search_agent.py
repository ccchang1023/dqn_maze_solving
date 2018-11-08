import numpy as np
import os, random, copy,math
from collections import deque
from heapq import heappush, heappop
from maze import generate_robot_map, TEST_MAZE, DEFAULT_MAZE
import numpy as np
import matplotlib.pyplot as plt


class NODE(object):
    def __init__(self, pos, move_count=0):
        self.pos = pos
        self.move_count = move_count
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
        print(search_params)
        self.algorithm = search_params.get('algorithm', "bfs")
        self.depth = search_params.get('depth', None)
        self.maze_type = search_params.get('maze_type', "10x10")
        self.path = deque()
        self.answer_path = deque()

        # self.maze = generate_robot_map()
        self.maze = DEFAULT_MAZE
        # self.maze = TEST_MAZE

        print(self.maze)
        nrows, ncols = np.shape(self.maze)
        self.pos = [0,0]
        self.goal = [0,ncols-1]
        self.visited_table = np.zeros([nrows, ncols], dtype=int)
        x,y = self.pos
        self.visited_table[x,y] = 1


        self.fig = plt.figure()
        plt.grid(True)
        nrows, ncols = np.shape(self.maze)
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])


    def search(self):

        root = NODE(self.pos)

        if self.algorithm == "dfs":
            if self.dfs(root, self.depth) == True:
                self.create_img(self.answer_path)
            else:
                print("Can't not find the find")


        elif self.algorithm == 'bfs':
            self.bfs()


        elif self.algorithm == 'Astar':
            self.Astar(root)

        return

    def get_successor(self, node, vt, move_count=0):
        successor = []
        x,y = node.pos
        next_state = [[x,y-1],[x-1,y],[x,y+1],[x+1,y]]
        for (xn, yn) in next_state:
            if self.is_valid(xn,yn) and not self.is_block(xn,yn) and vt[xn,yn]==0:
                n = NODE([xn,yn], move_count+1)
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

    def bfs(self):
        return

    def Astar(self, node):

        heap = []
        heuristic = self.euclidean_distance(node.pos, self.goal)
        node.set_heuristic(heuristic)
        heappush(heap,node)
        farest_dist = 0

        while len(heap) != 0:
            node = heappop(heap)

            # print("node pos:", node.pos)
            # print("node heuristic:", node.heuristic)
            # print("-------------------")

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
                heuristic = n.move_count + self.euclidean_distance(n.pos, self.goal)
                n.set_heuristic(heuristic)
                heappush(heap,n)

        return False


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


















