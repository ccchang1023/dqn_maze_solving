from __future__ import print_function
import os, sys, time, datetime, json, random
from maze import Maze, DIR
import numpy as np
import matplotlib.pyplot as plt
from maze import Maze, DEFAULT_MAZE
from keras.models import Sequential
from keras.layers.core import Dense


class test(object):
    def __init__(self):
        return
    def reset(self):
        self.a = "n"


def main():

    t = test()
    t.reset()
    return 

    m = Maze()  
    s,_,_,_ = m.move(DIR(random.randint(0,3)))
    print(s)
    
    list = []
    for i in range(100):
        list.append(i)
    for i,j in enumerate(np.random.choice(list, 5, replace=False)):
        print(i," ",j)
    

    
    envstate = s.reshape((1, -1))
    model = Sequential()
    model.add(Dense(32, input_shape=(s.size,1)))
    
    
    
if __name__ == '__main__':
    main()