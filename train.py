from __future__ import print_function
from search_agent import SEARCH_AGENT
from dqn import DQN
from ddqn import DDQN
from dueldqn import DUELDQN
from rrt_agent import RRT
import time

train_params={
    'batch_size' : 128,
    'gamma' : .95, #discount value when update the qvalue, 0~1
    'epsilon' : .05, #epsilon greedy for choosing best move, (the prob to choice the random move)
    'learning_rate' : 5e-4,
    'epochs' : 1000000,
    'num_moves_limit' : 100,
    'rounds_to_test' : 100,
    # 'load_maze_path' : "40x40Maze_98%",
    'saved_model_path' : "./saved_model/test.h5",
    # 'load_model_path' : "./saved_model/ddqn_10k.h5",
    'rounds_to_save_model' : 20000,
    'rounds_to_decay_lr' : 20000,
    'step_to_update_tModel' : 500,
    'maze_reward_lower_bound' : -0.03*1600,
    'db_capacity': 100000,
    #'tensorboard_log_path' : './log/test/',
    'Model_type': "dense",
}

search_params={
    'algorithm' : "Astar",
    'depth' : 40,
    "maze_type" : "10x10",
}

def main():

    # dqn = DQN(**train_params)
    # initial_rounds = 200
    # print ("Initial dataset:", initial_rounds, " rounds")
    # dqn.initial_dataset(initial_rounds)
    # print("Start training")
    # dqn.train()

    # ddqn = DDQN(**train_params)
    # initial_rounds = 200
    # print ("Initial dataset:", initial_rounds, " rounds")
    # ddqn.initial_dataset(initial_rounds)
    # print("Start training")
    # ddqn.train()

    # sa = SEARCH_AGENT(**search_params)
    # sa.search()

    # dqn = DUELDQN(**train_params)
    # initial_rounds = 300
    # print ("Initial dataset:", initial_rounds, " rounds")
    # dqn.initial_dataset(initial_rounds)
    # print("Start training")
    # dqn.train()

    rrt = RRT()
    s = time.clock()
    rounds = 100
    for _ in range(rounds):
        print(rrt.gen_path())
        rrt.reset()
    t = time.clock()
    print((t-s)/rounds)


    
if __name__ == "__main__":
    main()
