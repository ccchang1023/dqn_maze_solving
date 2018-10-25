from __future__ import print_function
# from dqn import DQN
# from ddqn import DDQN
from search_agent import SEARCH_AGENT

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
    # 'load_model_path' : "./saved_model/ddqn_10k.h5",
    'rounds_to_save_model' : 10000,
    'rounds_to_decay_lr' : 10000,
    'step_to_update_tModel' : 500,
    'maze_reward_lower_bound' : -0.03*1600,
    'db_capacity': 2000,
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
    # initial_rounds = 300
    # print ("Initial dataset:", initial_rounds, " rounds")
    # dqn.initial_dataset(initial_rounds)
    # print("Start training")
    # dqn.train()

    # ddqn = DDQN(**train_params)
    # initial_rounds = 300
    # print ("Initial dataset:", initial_rounds, " rounds")
    # ddqn.initial_dataset(initial_rounds)
    # print("Start training")
    # ddqn.train()

    sa = SEARCH_AGENT(**search_params)
    sa.search()




    
if __name__ == "__main__":
    main()
