from __future__ import print_function
import os, sys, time, datetime, json, random
from maze import Maze, DIR
from ddqn import DDQN
import numpy as np
import matplotlib.pyplot as plt
from maze import Maze, DEFAULT_MAZE
# from keras.models import Sequential
# from keras.layers.core import Dense
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    # 'load_model_path' : "./saved_model/40x40_lr5e-5_98%.h5",
    'rounds_to_save_model' : 10000,
    'rounds_to_decay_lr' : 10000,
    'maze_reward_lower_bound' : -0.03*1600,
    'db_capacity': 2000,
    #'tensorboard_log_path' : './log/test/',
    'Model_type': "dense",
}



def main():
    m = Maze()
    print(m.get_state())
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