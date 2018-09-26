from __future__ import print_function
import os, sys, time, datetime, json, random
from maze import Maze, DIR
import numpy as np
import matplotlib.pyplot as plt
from experience_db import ExperienceDB
from model import default_model, restore_model
from dqn import DQN


train_params={
    'batch_size' : 32,
    'gamma' : 1., #discount value when update the qvalue, 0~1
    'epsilon' : 0.05, #epsilon greedy for choosing best move, (the prob to choice the random move)
    'epochs' : 40000,
    'num_moves_limit' : 500,
    'rounds_to_test' : 100,
    'saved_model_path' : "./saved_model/test.h5",
    'rounds_to_save_model' : 20000
}

num_of_actions = 4
experience_db_capacity = 1000


def main():
    
    m = Maze()
    state_size = m.get_state().size
    model = default_model(state_size, num_of_actions)
    # model = restore_model('./saved_model/test.h5')
    e_db = ExperienceDB(model, experience_db_capacity)
    dqn = DQN(m, model, e_db, **train_params)
    n_rounds = 100
    print ("Initial dataset:", n_rounds, " rounds")
    dqn.initial_dataset(n_rounds)
    print("Start training")
    dqn.train()
    
    # m.show_animate()
    
    
    
if __name__ == "__main__":
    main()
