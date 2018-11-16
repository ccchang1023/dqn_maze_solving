import random, sys, os
import numpy as np
from maze import Maze, DIR
from model import default_model, deep_model, conv2d_model, dueldqn_model
from experience_db import ExperienceDB
from search_agent import SEARCH_AGENT
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.models import load_model
import global_setting as gl



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
        self.load_maze_path = train_params.get('load_maze_path', "")
        self.saved_model_path = train_params.get('saved_model_path', "")
        self.load_model_path = train_params.get('load_model_path', "")
        self.tensorboard_log_path = train_params.get("tensorboard_log_path", "")
        self.rounds_to_decay_lr = train_params.get("rounds_to_decay_lr", None)
        self.rounds_to_save_model = train_params.get('rounds_to_save_model', 10000)

        ######DQN parameters#####
        lb = train_params.get("maze_reward_lower_bound", None)
        self.maze = Maze(lower_bound = lb, load_maze_path = self.load_maze_path)
        self.db_capacity = train_params.get('db_capacity', 1000)
        if self.load_model_path != "":
            gl.set_model(load_model(self.load_model_path))
        else:
            #For conv2d
                # input_shape = np.shape(self.maze.get_state())
            # gl.set_model(conv2d_model(self.learning_rate, input_shape, self.maze.get_num_of_actions()))
            # gl.set_model(default_model(self.learning_rate, self.maze.get_state().size, self.maze.get_num_of_actions()))
            gl.set_model(deep_model(self.learning_rate, self.maze.get_state().size, self.maze.get_num_of_actions()))
            # gl.set_model(dueldqn_model(self.learning_rate, self.maze.get_state().size, self.maze.get_num_of_actions()))
        self.experience_db = ExperienceDB(db_cpacity = self.db_capacity, state_size=self.maze.get_state().size)
        self.sa = SEARCH_AGENT()
        self.sa.set_maze(maze=self.maze.maze)


    def initial_dataset(self, n_rounds):
        for _ in range(n_rounds):
            self.maze.reset()
            s = self.maze.get_state()
            d = np.argmax(gl.get_model().predict(s))
            s_next, r, is_goal, is_terminate = self.maze.move(DIR(d))
            transition = [s,d,r,s_next,is_terminate]
            self.experience_db.add_data(transition)

    def initial_opt_dataset(self, n_rounds):
        for _ in range(n_rounds):
            self.maze.reset()
            pos_list, dir_list = self.maze.get_opt_path2()
            for dir in dir_list:
                s = self.maze.get_state()
                s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                transition = [s,dir,r,s_next,is_terminate]
                self.experience_db.add_data(transition)

    #retrun the action of max Qvalue(predict by model)
    def get_best_action(self, state):
        return np.argmax(gl.get_model().predict(state)) #Return dir of max Qvalue
        
    def decay_learning_rate(self, decay=0.2):
        lr = K.get_value(gl.get_model().optimizer.lr)
        K.set_value(gl.get_model().optimizer.lr, lr*decay)
    
    def train(self):
        cycles = 16
        winrate_sum = prev_winrate_sum = 0.

        for i in range(self.epochs):
            # print("Epoch:%d Cycle:%d" %(i,j))
            tmp_count = 0
            for j in range(cycles):
                # print("Epoch:%d Cycle:%d" %(i,j))
                self.maze.reset()
                #====================Normal DQN====================
                # print("Normal DQN...")
                prev_pos = None
                for k in range(self.num_moves_limit):
                    s = self.maze.get_state()
                    if random.random() <= 0.8:  #move by solution
                        pos_list, dir_list = self.sa.search(start_pos=self.maze.token_pos, goal=self.maze.goal)
                        self.sa.reset()
                        dir = dir_list[0]
                    else:                       #move by randomness
                        if random.random() <= 0.5:
                            dir = np.random.randint(0, 3)
                        else:
                            dir = self.get_best_action(s)
                    s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                    transition = [s, dir, r, s_next, is_terminate]
                    self.experience_db.add_data(transition)

                    # Train model
                    inputs, answers = self.experience_db.get_data(self.batch_size, self.gamma)
                    # history = gl.get_model().fit(inputs, answers, epochs=1, batch_size =optimization_num, verbose=0)
                    train_loss = gl.get_model().train_on_batch(inputs, answers)

                    if is_terminate:
                        break

            if i % 10 == 0:
                sys.stdout.write("Epochs:%d" % (i))
                winrate_sum += self.test(self.rounds_to_test)

            # Decay learning_rate
            if i%200 == 0 and i != 0:
                if winrate_sum <= prev_winrate_sum:
                    self.decay_learning_rate(decay=0.5)
                    if K.get_value(gl.get_model().optimizer.lr) <= 1e-20:
                        print("Train Finish")
                        return
                    print("Decay learning rate to:", K.get_value(gl.get_model().optimizer.lr))
                prev_winrate_sum = winrate_sum
                winrate_sum = 0.

            # if i % self.rounds_to_decay_lr == 0 and i != 0:
            #     self.decay_learning_rate()
            #     print("Decay learning rate to:", K.get_value(gl.get_model().optimizer.lr))

            if self.rounds_to_save_model != 0 and i % self.rounds_to_save_model == 0:
                gl.get_model().save(self.saved_model_path)
                gl.get_model().save(self.saved_model_path[:-3] + "_target" + self.saved_model_path[-3:])



    def test(self, rounds=100, is_count_opt=False):
        win_count = 0.
        average_reward = 0.
        optimal_rate = 0.
        diff_count_sum = 0
        test_input = list()
        test_answer = list()
        moves_count = 0.
        goal_moves = 0.
        fail_moves = 0.
        average_goal_moves = average_fail_moves = 0

        for i in range(rounds):
            self.maze.reset()
            prev2_pos = None
            moves_count = 0.
            for j in range(self.num_moves_limit):
                s = self.maze.get_state()
                prev_pos = self.maze.token_pos.copy()
                # dir = self.get_best_action(s)
                a = gl.get_model().predict(s)    #With shape (batch, num_actions) -> (1,4)
                dir = np.argmax(a)

                s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                # self.maze.create_img()

                average_reward += r
                moves_count += 1

                #prevent from stucking in fail loop
                if(self.maze.token_pos == prev2_pos):
                    fail_moves += moves_count
                    break
                prev2_pos = prev_pos.copy()

                #Get corresponding x_test and y_test
                if is_terminate:
                    a[0][dir] = r + self.gamma*np.max(gl.get_model().predict(s_next))
                else:
                    a[0][dir] = r
                test_input.append(s)
                test_answer.append(a)

                if is_goal:
                    win_count += 1
                    goal_moves += moves_count
                    if is_count_opt:
                        diff_count = self.maze.get_optimal_solution_diff()
                        if diff_count == 0:
                            optimal_rate += 1
                        else:
                            diff_count_sum += diff_count
                    break

                if is_terminate or j==self.num_moves_limit-1 or self.maze.get_reward_sum() < self.maze.get_reward_lower_bound():
                    fail_moves += moves_count
                    break
            # self.maze.show_animate()

        test_input = np.squeeze(np.array(test_input), axis=1) #Transfer shape from [batch, 1, pixels] to [batch, pixels]
        test_answer  = np.squeeze(np.array(test_answer), axis=1) # Transfer shape from [batch, 1, 4] to [batch, 4]
        # print(np.shape(test_input), "   ", np.shape(test_answer))

        loss = gl.get_model().evaluate(np.array(test_input), np.array(test_answer), verbose=0)
        win_rate = (win_count/rounds)*100
        average_reward /= rounds
        if win_count !=0:
            average_goal_moves = (goal_moves/win_count)
        if (rounds-win_count)!=0:
            average_fail_moves = (fail_moves/(rounds-win_count))

        # print("f:", fail_moves, " g" , goal_moves, " wincount:", win_count, " rounds:", rounds)


        if is_count_opt:
            optimal_rate = (optimal_rate/rounds)*100
            output_str = str(" Loss:%f\tWin_rate:%.2f%%\tOptimal_solution_rate:%.2f%%\t"
                             "Diff_count_sum:%d\tAverage gmoves:%f\tAverage fmoves:%f\tAverage_reward:%.4f"
                             %(loss, win_rate, optimal_rate, diff_count_sum, average_goal_moves, average_fail_moves, average_reward))

        else:
            output_str = str(" Loss:%f\tWin_rate:%.2f%%\tDiff_count_sum:%d\tAverage gmoves:%.2f\tAverage fmoves:%.2f\tAverage_reward:%.4f"
                             %(loss, win_rate, diff_count_sum, average_goal_moves, average_fail_moves,average_reward))
        print(output_str)
        
        return win_rate
