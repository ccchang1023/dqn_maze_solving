from __future__ import print_function
from dqn import DQN


train_params={
    'batch_size' : 32,
    'gamma' : 1., #discount value when update the qvalue, 0~1
    'epsilon' : 0.05, #epsilon greedy for choosing best move, (the prob to choice the random move)
    'learning_rate' : 1e-4,
    'epochs' : 100000,
    'num_moves_limit' : 500,
    'rounds_to_test' : 100,
    'saved_model_path' : "./saved_model/test.h5",
    # 'tensorboard_log_path' : './log/test/',
    'rounds_to_save_model' : 20000,
    'maze_reward_lower_bound' : -3.,
    'db_capacity': 2000,
    'Model_type': "dense",
}

def main():

    # model = restore_model('./saved_model/test.h5')
    dqn = DQN(**train_params)
    initial_rounds = 100
    print ("Initial dataset:", initial_rounds, " rounds")
    dqn.initial_dataset(initial_rounds)
    print("Start training")
    dqn.train()
    
    # m.show_animate()
    
    
    
if __name__ == "__main__":
    main()
