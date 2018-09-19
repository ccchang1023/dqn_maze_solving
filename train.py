from __future__ import print_function
import os, sys, time, datetime, json, random
from maze import MAZE, DIR
import numpy as np
import matplotlib.pyplot as plt



def main():
    # A = np.random.rand(5, 5)
    # plt.figure(1)
    # plt.imshow(A, interpolation='nearest')
    # plt.grid(True)
    
    # plt.figure(2)
    # plt.imshow(A, interpolation='bilinear')
    # plt.grid(True)
    
    # plt.figure(3)
    # plt.imshow(A, interpolation='bicubic')
    # plt.grid(True)
    # plt.show()
    # return
    
    m = MAZE()
    # m.reset()
    # print(random.choice(list(DIR)))
    
    for _ in range(5): 
        dir = random.randint(0,3)
        # dir = random.choice(list(DIR))
        is_valid, is_terminate, reward = m.move(dir)
        m.show()
        plt.show()
    
    
    
if __name__ == "__main__":
    main()