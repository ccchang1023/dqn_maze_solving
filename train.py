from __future__ import print_function
import os, sys, time, datetime, json, random
from maze import Maze, DIR
import numpy as np
import matplotlib.pyplot as plt
from experience_db import ExperienceDB
from model import default_model
from dqn import DQN


train_params={
    'batch_size' : 32,
    'gamma' : 0.9, #discount value when update the qvalue, 0~1
    'epsilon' : 0.1, #epsilon greedy for choosing best move, (the prob to choice the random move)
    'epochs' : 30000,
    'step_limit' : 50,
    'rounds_to_test' : 100,
    'checkpoint_file' : "",
}

num_of_actions = 4
experience_db_capacity = 1000


def main():
    
    m = Maze()
    # m.reset()
    # print(random.choice(list(DIR)))
    maze_size = m.get_state().size
    model = default_model(maze_size, num_of_actions)
    e_db = ExperienceDB(model, experience_db_capacity)
    dqn = DQN(m, model, e_db, **train_params)
    
    n_rounds = 500
    print ("Initial dataset:", n_rounds, " rounds")
    dqn.initial_dataset(n_rounds)
    print("Start training")
    dqn.train()
    
    
    # m.show_animate()
    
    
   
    
if __name__ == "__main__":
    main()