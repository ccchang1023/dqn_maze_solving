import random, sys, os
import numpy as np
from maze import Maze, DIR
from keras import backend as K
from keras.models import load_model, clone_model
from dqn import DQN
from search_agent import SEARCH_AGENT
import global_setting as gl
import time



class DDQN(DQN):

    def __init__(self, **train_params):
        super(DDQN, self).__init__(**train_params)
        # DQN.__init__(self, train_params)
        gl.init_targetModel()

    def initial_opt_dataset_by_SA(self, n_rounds):
        for _ in range(n_rounds):
            self.maze.reset()
            pos_list, dir_list = self.sa.search(start_pos=self.maze.token_pos, goal=self.maze.goal)
            self.sa.reset()
            for dir in dir_list:
                s = self.maze.get_state()
                s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                transition = [s,dir,r,s_next,is_terminate]
                self.experience_db.add_data(transition)
            if r != 1:
                print("Bug:Can't find path", r)
                input("Wait..")

    def update_target_model(self):
        gl.update_targetModel()

    def train(self):
        hindsight_batch_size = 4
        optimization_num = 40
        cycles = 16
        winrate_sum = prev_winrate_sum = 0.
        solution_dict = self.sa.load_solution_db('Sol_3d_robot_map.npy')

        for i in range(self.epochs):
            # print("Epoch:%d" %(i))
            # ts = time.clock()
            # move_sum = 0
            for j in range(cycles):
                # print("Last move count:", self.maze.get_move_count())
                # print("Epoch:%d Cycle:%d" %(i,j))
                self.maze.reset()
                # print("token_pos:", self.maze.token_pos)
                # print("goal:", self.maze.goal)
                transition_list = []
                origin_token_pos = self.maze.get_token_pos()
                origin_goal = self.maze.get_goal_pos()

                #====================Normal DQN====================
                # print("Normal DQN...")
                prev_pos = None
                for k in range(self.num_moves_limit):
                    # move_sum += 1
                    s = self.maze.get_state()
                    # print("Start:", self.maze.token_pos)
                    # print("Goal:", self.maze.goal)
                    # print("State:", s)

                    if random.random() <= 0.5:  #move by solution
                        # pos_list, dir_list = self.sa.search(start_pos=self.maze.token_pos, goal=self.maze.goal)
                        # self.sa.reset()

                        dir_list = solution_dict[tuple(self.maze.token_pos)]
                        # print(dir_list)
                        dir = dir_list[0]
                        if dir_list == None:
                            print("Bug : can't find goal from ", origin_token_pos, " to ", origin_goal)
                        # print("Follow solution:")
                        # print(pos_list)
                        # print(dir_list)
                    else:                       #move by randomness
                        if random.random() <= 0.5:
                            dir = np.random.randint(0, 3)
                        else:
                            dir = self.get_best_action(s)

                        # self.maze.create_img()
                        # if prev_pos != [s_next[0][0],s_next[0][1]]:
                        #     self.experience_db.add_data(transition)
                        #     # transition_list.append(transition)
                        # prev_pos = [s[0][0], s[0][1]]

                    s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                    transition = [s, dir, r, s_next, is_terminate]
                    self.experience_db.add_data(transition)

                    # Train model
                    inputs, answers = self.experience_db.get_data_by_ddqn(self.batch_size, self.gamma)
                    # history = gl.get_model().fit(inputs, answers, epochs=1, batch_size =optimization_num, verbose=0)
                    train_loss = gl.get_model().train_on_batch(inputs, answers)

                    if is_terminate:
                        break

            # te = time.clock()
            # print("Cylce average time:", (te-ts)/move_sum)

            # Update target model
            self.update_target_model()

            if i % 10 == 0:
                sys.stdout.write("Epochs:%d" % (i))
                winrate_sum += self.test(self.rounds_to_test)

            # Decay learning_rate
            if i%100 == 0 and i != 0:
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