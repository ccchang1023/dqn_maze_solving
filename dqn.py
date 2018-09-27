import random
from maze import DIR
import numpy as np
from keras import backend as K

class DQN(object):        
    def __init__(self, maze=None, model=None, experience_db=None, **params):
        self.maze = maze
        self.model = model
        self.experience_db = experience_db
        self.batch_size = params.get('batch_size', 32)
        self.gamma = params.get('gamma', 0.95)
        self.epsilon = params.get('epsilon', 0.)
        self.epochs = params.get('epochs', 100)
        self.num_moves_limit = params.get('num_moves_limit', None)
        self.rounds_to_test = params.get("rounds_to_test", 100)
        self.saved_model_path = params.get('saved_model_path', "")
        self.rounds_to_save_model = params.get('rounds_to_save_model', 0)
    
    def initial_dataset(self, n_rounds):
        for _ in range(n_rounds):
            dir = random.randint(0,3)
            s = self.maze.get_state()
            s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
            transition = [s,dir,r,s_next,is_terminate]
            self.experience_db.add(transition)
    
    #retrun the action of max Qvalue(predict by model)
    def get_best_action(self, state):
        return np.argmax(self.model.predict(state)) #Return dir of max Qvalue
        
    def decay_learning_rate(self, decay=0.2):
        lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, lr*decay)
    
    def train(self):
        loss_sum = 0.
        loss_sum_prev = 0.
        for i in range(self.epochs):
            print("Epoch:%d" %(i))
            self.maze.reset()
            
            #Decay learning_rate
            if i % 10000 == 0:
                self.decay_learning_rate()
            # if i%500 == 0:
                # if loss_sum_prev != 0. and loss_sum_prev < loss_sum:
                    # print(loss_sum_prev, "  ", loss_sum)
                    # self.decay_learning_rate()
                    # print("Decay learning rate to:",K.get_value(self.model.optimizer.lr))
                # loss_sum_prev = loss_sum
                # loss_sum = 0.
            # print("Epoch:", i)
            for j in range(self.num_moves_limit):
                s = self.maze.get_state()
                if random.random() > self.epsilon:
                    dir = self.get_best_action(s)
                else:
                    dir = np.random.randint(0,3)
                s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                transition = [s,dir,r,s_next,is_terminate]
                self.experience_db.add(transition)
                # self.maze.create_img()
                inputs, answers = self.experience_db.get_data(self.batch_size, self.gamma)
                history = self.model.fit(inputs, answers, epochs=8, batch_size=16, verbose=0)
                loss = self.model.evaluate(inputs, answers, verbose=0)
                loss_sum += loss
                
                #Even the game return terminate, keep training until reach goal or surpass lower bound 
                #(but update reward will only be r, check in get_data function)
                if is_goal or self.maze.get_reward_sum() < self.maze.get_reward_lower_bound():
                    break

            if i%50 == 0:
                print("Epoch:%d, move_count:%d, reward_sum:%f, loss:%f" %(i, self.maze.get_move_count(), 
                      self.maze.get_reward_sum(), loss))
            if i%self.rounds_to_test==0:
                self.test(self.rounds_to_test)

            if self.rounds_to_save_model != 0 and i%self.rounds_to_save_model == 0:
                self.model.save(self.saved_model_path)
            
    def test(self, rounds=100):
       win_rate = 0.
       average_reward = 0.
       loss = 0.
       optimal_rate = 0.
       diff_count_sum = 0
       for i in range(rounds):
           self.maze.reset()
           for j in range(self.num_moves_limit):
               s = self.maze.get_state()
               dir = self.get_best_action(s)
               _, r, is_goal, is_terminate = self.maze.move(DIR(dir))
               average_reward += r
               if is_goal:
                   win_rate += 1
                   diff_count = self.maze.get_optimal_solution_diff()
                   if diff_count == 0:
                       optimal_rate += 1
                   else:
                       diff_count_sum += diff_count
                       
               if is_terminate:
                   break
       
       inputs, answers = self.experience_db.get_data(self.batch_size, self.gamma)
       loss = self.model.evaluate(inputs, answers, verbose=0)
       win_rate = (win_rate/rounds)*100
       optimal_rate = (optimal_rate/rounds)*100
       average_reward /= rounds
       output_str = str("Test Result: Loss:%f   Win_rate:%.2f%%     Optimal_solution_rate:%.2f%%    "
                        "Diff_count_sum:%d      Average_reward:%.4f"
                        %(loss, win_rate, optimal_rate, diff_count_sum, average_reward))
       print(output_str)
        
        
