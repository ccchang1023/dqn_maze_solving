from __future__ import print_function
import os, sys, time, datetime, json, random
from maze import Maze, DIR
from search_agent import SEARCH_AGENT
from ddqn import DDQN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from mayavi import mlab

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


train_params={
    'batch_size' : 8,
    'gamma' : .95, #discount value when update the qvalue, 0~1
    'epsilon' : .05, #epsilon greedy for choosing best move, (the prob to choice the random move)
    'learning_rate' : 5e-5,
    'epochs' : 1000000,
    'num_moves_limit' : 400,
    'rounds_to_test' : 100,
    # 'load_maze_path' : "40x40Maze_98%",
    'saved_model_path' : "./saved_model/test.h5",
    'load_model_path' : "./saved_model/3d_ep2400_lr7.8e-7.h5",
    'rounds_to_save_model' : 10000,
    'rounds_to_decay_lr' : 10000,
    'maze_reward_lower_bound' : -0.03*1600,
    'db_capacity': 2000,
    #'tensorboard_log_path' : './log/test/',
    'Model_type': "dense",
}

search_params={
    'algorithm' : "Astar",
    'depth' : 45,
    "maze_type" : "10x10",
}



def main():

    ddqn = DDQN(**train_params)
    ddqn.test(100)
    return

    # ddqn.maze.create_img()
    # return
    # ddqn.maze.set_token_pos([39,0,0])
    # pos_list, dir_list = ddqn.sa.search(start_pos=ddqn.maze.token_pos, goal=ddqn.maze.goal)
    # ddqn.sa.reset()
    # for dir in dir_list:
    #     _,_,_,_ = ddqn.maze.move(DIR(dir))
    # ddqn.maze.create_img()
    # return

    # ddqn.sa.save_solution_db(goal=ddqn.maze.goal)
    dict = ddqn.sa.load_solution_db('Sol_3d_robot_map.npy')
    # return

    nrows, ncols, height = np.shape(ddqn.maze.maze)
    road_list = [[x, y, z] for x in range(nrows) for y in range(ncols) for z in range(height) if
                 ddqn.maze.maze[x][y][z] == 1]
    for pos in road_list:
        ddqn.maze.reset(pos)
        dir_list = dict[tuple(pos)]
        for dir in dir_list:
            s, r, gTag, tTag = ddqn.maze.move(DIR(dir))
        # print(r)
        if r != 1:
            print("Start s:", ddqn.maze.get_token_pos(), " g: ", ddqn.maze.get_goal_pos())
            input("Bug...")
    return

    for dir in dir_list:
        s,r,_,_ = ddqn.maze.move(DIR(dir))
        print("Curretn pos:",ddqn.maze.get_token_pos())
        x,y,z = ddqn.maze.get_token_pos()
        print(ddqn.maze.maze[x][y][z])
        # print(r, " ", s)

    # ddqn.maze.create_img()
    return


    sa = SEARCH_AGENT(**search_params)
    round = 10
    start = time.clock()
    for _ in range(round):
        # m.reset()
        # sa.set_maze(m.maze)
        sa.reset()
        s = random.choice(sa.road_list)
        g = random.choice(sa.road_list)
        # m.set_token_pos(s)
        # m.set_goal(g)
        pos_list, dir_list = sa.search(start_pos=s, goal=g)
        if pos_list==None:
            print("fail")
        # pos_list, dir_list = sa.search(start_pos=m.token_pos, goal=m.goal)

        # print("Start:",m.token_pos)
        # print("Goal:",m.goal)

        # print(pos_list)
        # print(dir_list)

        # for dir in dir_list:
        #     s ,r , gTag, tTag = m.move(DIR(dir))
        #     print([s ,r , gTag, tTag])

    end = time.clock()
    print("Average time:", (end-start)/round)
    return


    for _ in range(10000000):
        m.reset()
        return
        p = random.choice(m.road_list)
        m.set_token_pos(p)
        p,d = m.get_opt_path2()
        for i in d:
            s, r, gTag, tTag = m.move(DIR(i))
        if r != 1:
            print("Start s:", m.get_token_pos(), " g: ", m.get_goal_pos())
            input("Bug...")

    return

    while True:
        m.reset()
        m.show_maze()
        for i in range(300):
            dir = input("Enter dir:")
            os.system('clear')
            # dir = np.random.randint(0,3)
            s ,r , gTag, tTag = m.move(DIR(int(dir)))
            print("reward:%f, valid_tag:%d, terminal_tag:%d" %(r, gTag, tTag))
            # m.show_maze()
            print(m.get_state().reshape(10,10))
            if tTag:
                print("Dead!")
                break
        # print(m.get_state().reshape(10,10))
        m.get_optimal_solution_diff()
        return
            
    # m = Maze()  
    # s,_,_,_ = m.move(DIR(random.randint(0,3)))
    # print(s)
    
    # list = []
    # for i in range(100):
        # list.append(i)
    # for i,j in enumerate(np.random.choice(list, 5, replace=False)):
        # print(i," ",j)
    
    # envstate = s.reshape((1, -1))
    # model = Sequential()
    # model.add(Dense(32, input_shape=(s.size,1)))
    
    
    
if __name__ == '__main__':
    main()