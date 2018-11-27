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
        self.maze = generate_robot_map(size=40)
        # self.maze = DEFAULT_MAZE
        # self.maze = TEST_MAZE
        # print(self.maze)
        self.nrows, self.ncols, self.height = np.shape(self.maze)
        # self.road_list = [[x, y] for x in range(self.nrows) for y in range(self.ncols) if self.maze[x, y] == 1]
        self.reset()

    def set_maze(self, maze):
        self.maze = maze
        self.nrows, self.ncols, self.height = np.shape(self.maze)
        self.visited_table = np.zeros([self.nrows, self.ncols, self.height], dtype=int)
        # self.road_list = [[x, y] for x in range(self.nrows) for y in range(self.ncols) if self.maze[x, y] == 1]

    def reset(self):
        self.visited_table = np.zeros([self.nrows, self.ncols, self.height], dtype=int)


    def search(self, start_pos=None, goal = None):

        root = NODE(start_pos)
        x,y,z = start_pos
        self.visited_table[x][y][z] = 1
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
        x,y,z = node.pos
        next_state = [[x,y-1,z],[x-1,y,z],[x,y+1,z],[x+1,y,z],[x,y,z+1],[x,y,z-1]]
        for i in range(len(next_state)):
            xn,yn,zn = next_state[i]
            if self.is_valid(xn,yn,zn) and not self.is_block(xn,yn,zn) and self.visited_table[xn,yn,zn]==0:
                self.visited_table[xn, yn, zn] = 1
                n = NODE(pos=[xn,yn,zn], move_count=move_count+1, pre_dir=i)
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
        return math.sqrt(abs(p1[0]-p2[0])**2 + abs(p1[1]-p2[1])**2 + abs(p1[2]-p2[2])**2)


    def is_block(self, x,y,z):
        return True if self.maze[x,y,z] == 0 else False

    def is_valid(self,x,y,z):
        r,c,h = np.shape(self.maze)
        return True if x>=0 and x<r and y >=0 and y<c and z>=0 and z<h else False


    def save_solution_db(self, goal):
        nrows, ncols, height = np.shape(self.maze)
        road_list = [[x, y, z] for x in range(nrows) for y in range(ncols) for z in range(height) if
                         self.maze[x][y][z] == 1]
        dict = {}
        for pos in road_list:
            self.reset()
            print("Start from ", pos)
            pos_list, dir_list = self.search(pos, goal)
            if dir_list==None:
                print("Bug: Can't find path")
            dict[tuple(pos)] = dir_list
        # Save
        np.save('Sol_3d_robot_map.npy', dict)


    def load_solution_db(self, path):
        dict = np.load(path).item()
        if len(dict) == 0:
            print("Load db failed!!")
        return dict
        # nrows, ncols, height = np.shape(self.maze)
        # road_list = [[x, y, z] for x in range(nrows) for y in range(ncols) for z in range(height) if
        #              self.maze[x][y][z] == 1]
        # for pos in road_list:
        #     print(dict[tuple(pos)])
        #     if pos[2] == 39:
        #         break



    def create_img(self, answer):
        #reset
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

        # plot cube
        self.ax = self.fig.gca(projection='3d')
        self.ax.voxels(np.where(self.maze == 0, 1, 0), edgecolors='gray')
        # plot token pos
        x, y, z = self.token_pos
        plt.plot([x],[y],[z], marker='o', markersize=3, color="red")
        #plot goal pos
        x, y, z = self.goal
        plt.plot([x],[y],[z], marker='o', markersize=3, color="green")
        # plot visited path
        nrows, ncols, height = np.shape(self.maze)
        visited_point = [[x,y,z] for x in range(nrows) for y in range(ncols) for z in range(height) if self.visited_list[x][y][z]==1]
        for x,y,z in visited_point:
            plt.plot([x], [y], [z], marker='o', markersize=3, color="gray")


    # def show_animate(self):
    #     ani = animation.ArtistAnimation(self.fig, self.img_list, interval=50, blit=True, repeat_delay=1000)
    #     plt.show()
        # ani.save("test.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
        # ani.save("loop.mp4")


















