import random, sys
import numpy as np
from maze import Maze, DIR
from model import default_model
from experience_db import ExperienceDB
from keras import backend as K


class DQN(object):        
    def __init__(self, **train_params):
        print(train_params)
        #####Training parameters#####
        self.batch_size = train_params.get('batch_size', 32)
        self.learning_rate  = train_params.get('learning_rate', 1e-4)
        self.gamma = train_params.get('gamma', 1.)
        self.epsilon = train_params.get('epsilon', 0.05)
        self.epochs = train_params.get('epochs', 100)
        self.num_moves_limit = train_params.get('num_moves_limit', None)
        self.rounds_to_test = train_params.get("rounds_to_test", 100)
        self.saved_model_path = train_params.get('saved_model_path', "")
        self.rounds_to_save_model = train_params.get('rounds_to_save_model', 10000)

        ######DQN parameters#####
        lb = train_params.get("maze_reward_lower_bound", None)
        self.maze = Maze(lower_bound = lb)
        self.db_capacity = train_params.get('db_capacity', 1000)
        self.model = default_model(self.learning_rate, self.maze.get_state().size, self.maze.get_num_of_actions())
        self.experience_db = ExperienceDB(self.model, self.db_capacity)


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
            self.maze.reset(fix_goal=False)
            # print("Epoch:%d" %(i))

            # Decay learning_rate
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
            keep_playing = False
            for j in range(self.num_moves_limit):
                s = self.maze.get_state()
                if keep_playing or random.random() <= self.epsilon:
                    dir = np.random.randint(0, 3)
                    keep_playing = False
                else:
                    dir = self.get_best_action(s)
                s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                transition = [s,dir,r,s_next,is_terminate]
                self.experience_db.add(transition)
                # self.maze.create_img()
                inputs, answers = self.experience_db.get_data(self.batch_size, self.gamma)
                history = self.model.fit(inputs, answers, epochs=8, batch_size =16, verbose=0)
                loss = self.model.evaluate(inputs, answers, verbose=0)
                loss_sum += loss

                # if is_terminate:
                #     break
                #Even the game return terminate, keep training until reach goal or surpass lower bound
                if is_goal or self.maze.get_reward_sum() < self.maze.get_reward_lower_bound():
                    break
                elif is_terminate:
                    keep_playing = True


            # if i%100 == 0:
            #     print("Epoch:%d, move_count:%d, reward_sum:%f, loss:%f" %(i, self.maze.get_move_count(),
            #           self.maze.get_reward_sum(), loss))
            if i % 50 == 0:
                sys.stdout.write("Epochs:%d" %(i))
                self.test(self.rounds_to_test)

            if self.rounds_to_save_model != 0 and i%self.rounds_to_save_model == 0:
                self.model.save(self.saved_model_path)
            
    def test(self, rounds=100):
       win_rate = 0.
       average_reward = 0.
       optimal_rate = 0.
       diff_count_sum = 0
       test_input = list()
       test_answer = list()

       for i in range(rounds):
           self.maze.reset(fix_goal=True)
           for j in range(self.num_moves_limit):
               s = self.maze.get_state()
               # dir = self.get_best_action(s)
               a = self.model.predict(s)    #With shape (batch, num_actions) -> (1,4)
               dir = np.argmax(a)

               s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
               average_reward += r

               #Get corresponding x_test and y_test
               if is_terminate:
                   a[0][dir] = r + self.gamma*np.max(self.model.predict(s_next))
               else:
                   a[0][dir] = r
               test_input.append(s)
               test_answer.append(a)

               if is_goal:
                   win_rate += 1
                   diff_count = self.maze.get_optimal_solution_diff()
                   if diff_count == 0:
                       optimal_rate += 1
                   else:
                       diff_count_sum += diff_count
               if is_terminate:
                   break

       test_input = np.squeeze(np.array(test_input), axis=1) #Transfer shape from [batch, 1, pixels] to [batch, pixels]
       test_answer  = np.squeeze(np.array(test_answer), axis=1) # Transfer shape from [batch, 1, 4] to [batch, 4]
       # print(np.shape(test_input), "   ", np.shape(test_answer))

       loss = self.model.evaluate(np.array(test_input), np.array(test_answer), verbose=0)
       win_rate = (win_rate/rounds)*100
       optimal_rate = (optimal_rate/rounds)*100
       average_reward /= rounds
       output_str = str(" Loss:%f   Win_rate:%.2f%%     Optimal_solution_rate:%.2f%%    "
                        "Diff_count_sum:%d      Average_reward:%.4f"

                        %(loss, win_rate, optimal_rate, diff_count_sum, average_reward))
       print(output_str)
        
        
