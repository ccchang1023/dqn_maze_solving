from __future__ import print_function
import os, sys, time, datetime, json, random
from maze import Maze, DIR
import numpy as np
import matplotlib.pyplot as plt
from maze import Maze, DEFAULT_MAZE
# from keras.models import Sequential
# from keras.layers.core import Dense


def main():

    m = Maze()    
    print(np.random.choice(DIR))
    return
    
    while True:
        m.reset()
        print(m.get_state().reshape(9,10))
        for i in range(300):
            dir = input("Enter dir:")
            os.system('clear')
            s ,r , gTag, tTag = m.move(DIR(dir))
            print("reward:%f, valid_tag:%d, terminal_tag:%d" %(r, gTag, tTag))
            print("TOken:", m.get_token_pos())
            print(m.get_state().reshape(9,10))
            if tTag:
                print("Dead!")
                break
            
            
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