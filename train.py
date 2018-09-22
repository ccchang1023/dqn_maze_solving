from __future__ import print_function
import os, sys, time, datetime, json, random
from maze import Maze, DIR
import numpy as np
import matplotlib.pyplot as plt
from experience_db import ExperienceDB
from model import default_model
from dqn import DQN


train_params={
    'batch_size' : 5,
    'gamma' : 0.95,
    'epsilon' : 0.,
    'n_epoch' : 100,
    'batch_size' : 10,
    'checkpoint_file' : "",
}


def main():
    
    m = Maze()
    # m.reset()
    # print(random.choice(list(DIR)))
    maze_size = m.get_state().size
    num_of_actions = 4
    model = default_model(maze_size, num_of_actions)
    e_db = ExperienceDB(model)
    dqn = DQN(m, model, e_db, **train_params)
    
    n_rounds = 10
    dqn.initial_dataset(n_rounds)
    dqn.train()
    
    
    
    # m.show_animate()
    
    
   
    
if __name__ == "__main__":
    main()