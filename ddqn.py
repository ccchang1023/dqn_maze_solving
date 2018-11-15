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
        self.step_to_update_tModel = train_params.get("step_to_update_tModel", 100)
        self.sa = SEARCH_AGENT()
        self.sa.set_maze(maze=self.maze.maze)

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
        cycles = 16
        optimization_num = 40
        winrate_sum = prev_winrate_sum = 0.


        for i in range(self.epochs):
            # print("Epoch:%d Cycle:%d" %(i,j))
            tmp_count = 0
            for j in range(cycles):
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
                    s = self.maze.get_state()
                    if random.random() <= self.epsilon:
                        dir = np.random.randint(0, 3)
                    else:
                        dir = self.get_best_action(s)
                    # self.maze.create_img()
                    s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                    transition = [s, dir, r, s_next, is_terminate]

                    if prev_pos != [s_next[0][0],s_next[0][1]]:
                        self.experience_db.add_data(transition)
                        # transition_list.append(transition)

                    prev_pos = [s[0][0], s[0][1]]

                    if is_terminate or self.maze.get_reward_sum() < self.maze.get_reward_lower_bound():
                        break

                # tmp_count += self.maze.move_count
                # if j == cycles-1:
                #     print("Average move count:", tmp_count / cycles)
                #     self.maze.show_animate()

                # print("transition_list:")
                # print(transition_list)
                # a = [transition_list[0][0][0][0],transition_list[0][0][0][1]]
                # print(a)
                # print(np.shape(a))

                # ====================Add solution data====================
                # print("Add solution data...")
                self.maze.reset()
                self.maze.set_token_pos(origin_token_pos)
                self.maze.set_goal(origin_goal)
                # print("token_pos2:", self.maze.token_pos)
                # print("goal2:", self.maze.goal)
                # pos_list, dir_list = self.maze.get_opt_path2()
                pos_list, dir_list = self.sa.search(start_pos=self.maze.token_pos, goal=self.maze.goal)
                self.sa.reset()
                # print(pos_list)
                # print(dir_list)
                #complete path
                for k in range(len(dir_list)):
                    s = self.maze.get_state()
                    dir = dir_list[k]
                    s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                    transition = [s, dir, r, s_next, is_terminate]
                    # print(transition)
                    # self.experience_db.add_data(transition)
                    transition_list.append(transition)
                #random batch
                # num = min(len(pos_list), hindsight_batch_size)
                # for k in np.random.choice(len(pos_list), num, replace=False):
                #     self.maze.reset()
                #     if k == 0:
                #         self.maze.set_token_pos(origin_token_pos)
                #     else:
                #         self.maze.set_token_pos(pos_list[k - 1])
                #     self.maze.set_goal(origin_goal)
                #     s = self.maze.get_state()
                #     dir = dir_list[k]
                #     s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                #     transition = [s, dir, r, s_next, is_terminate]
                #     self.experience_db.add_data(transition)

                if r != 1:
                    print("Bug:Can't find path",r)
                    print("token_pos:", origin_token_pos)
                    print("goal:", origin_goal)
                    input("Wait..")


                # ====================Hindsight experience replay====================
                self.maze.reset()
                self.maze.set_token_pos(origin_token_pos)
                self.maze.set_goal(origin_goal)
                # print("token_pos3:", self.maze.token_pos)
                # print("goal3:", self.maze.goal)
                # print("HER...")
                # print("transition list:", transition_list)
                for k in range(len(transition_list)-1):
                    pos = [transition_list[k][0][0][0],transition_list[k][0][0][1]]  #get token_pos
                    num = min(len(transition_list)-k-1,hindsight_batch_size)
                    # print("----next trans----")
                    for m in np.random.choice(len(transition_list)-k-1, num, replace=False):
                        self.maze.reset()
                        self.maze.set_token_pos(pos)
                        # print("virtual token_pos:", self.maze.token_pos)

                        virtual_goal = [transition_list[k+m+1][0][0][0],transition_list[k+m+1][0][0][1]]   #get goal
                        self.maze.set_goal(virtual_goal)
                        # print("virtual goal_pos:", self.maze.goal)
                        s = self.maze.get_state()

                        dir = transition_list[k][1]   #get dir
                        # print("dir:",dir)

                        s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                        transition = [s, dir, r, s_next, is_terminate]
                        # print("sol transition:", transition)
                        self.experience_db.add_data(transition)
                    # input("Wait..")

                self.maze.set_token_pos(origin_token_pos)
                self.maze.set_goal(origin_goal)


            #Train model
            # start = time.clock()
            for _ in range(optimization_num):
                inputs, answers = self.experience_db.get_data_by_ddqn(self.batch_size, self.gamma)
                # history = gl.get_model().fit(inputs, answers, epochs=1, batch_size =optimization_num, verbose=0)
                train_loss = gl.get_model().train_on_batch(inputs, answers)

            # end = time.clock()
            # print("Time:", (end-start))

            # Update target model
            self.update_target_model()

            # Decay learning_rate
            if i%200 == 0 and i != 0:
                if winrate_sum <= prev_winrate_sum:
                    self.decay_learning_rate(decay=0.5)
                    print("Decay learning rate to:", K.get_value(gl.get_model().optimizer.lr))
                prev_winrate_sum = winrate_sum
                winrate_sum = 0.

            # if i % self.rounds_to_decay_lr == 0 and i != 0:
            #     self.decay_learning_rate()
            #     print("Decay learning rate to:", K.get_value(gl.get_model().optimizer.lr))

            if i % 10 == 0:
                sys.stdout.write("Epochs:%d" % (i))
                winrate_sum += self.test(self.rounds_to_test)

            if self.rounds_to_save_model != 0 and i % self.rounds_to_save_model == 0:
                gl.get_model().save(self.saved_model_path)
                gl.get_model().save(self.saved_model_path[:-3] + "_target" + self.saved_model_path[-3:])