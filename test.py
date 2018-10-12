from __future__ import print_function
import os, sys, time, datetime, json, random
from maze import Maze, DIR
import numpy as np
import matplotlib.pyplot as plt
from maze import Maze, DEFAULT_MAZE
# from keras.models import Sequential
# from keras.layers.core import Dense
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():




    # x = np.arange(0, np.pi * 2, np.pi / 10.0)
    # y = np.sin(x)
    # fig = plt.figure()
    # imgs = []
    # for i in range(len(x)):
    #     img = plt.plot(x[:i + 1], y[:i + 1], 'b-o')
    #     imgs.append(img)
    # anim = animation.ArtistAnimation(fig, imgs, interval=100)
    # anim.save('result1.gif', 'imagemagick')
    # plt.show()
    # return


    m = Maze()
    m.create_img()

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