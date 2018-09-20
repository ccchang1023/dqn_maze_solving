from __future__ import print_function
import os, sys, time, datetime, json, random
from maze import Maze, DIR
import numpy as np
import matplotlib.pyplot as plt
from experience_db import ExperienceDB
from model import default_model


def main():
    
    m = Maze()
    # m.reset()
    # print(random.choice(list(DIR)))
    maze_size = m.get_state().size
    num_of_actions = 4
    
    model = default_model(maze_size, num_of_actions)
    experience_db = ExperienceDB(model)
    
    
    for _ in range(5):
        dir = random.randint(0,3)
        # dir = random.choice(list(DIR))
        s = m.get_state()
        s_next, r, is_valid, is_terminate = m.move(DIR(dir))
        
        transition = [s,dir,r,s_next,is_terminate]
        # print(np.shape(transition))
        experience_db.add(transition)
        
        
        # m.create_img()
    
    # experience_db.show_data()
    batch_size = 5
    gamma = 0.95
    inputs, answers = experience_db.get_data(batch_size, gamma)

    
    
    
    # m.show_animate()
    
    
   
    
if __name__ == "__main__":
    main()