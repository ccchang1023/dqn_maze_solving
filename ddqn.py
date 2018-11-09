import random, sys, os
import numpy as np
from maze import Maze, DIR
from keras import backend as K
from keras.models import load_model, clone_model
from dqn import DQN
import global_setting as gl


class DDQN(DQN):

    def __init__(self, **train_params):
        super(DDQN, self).__init__(**train_params)
        # DQN.__init__(self, train_params)
        gl.init_targetModel()
        self.step_to_update_tModel = train_params.get("step_to_update_tModel", 100)

    def update_target_model(self):
        gl.update_targetModel()

    def train(self):
        hindsight_batch_size = 4
        episode_num = 16
        optimization_num = 40
        winrate_sum = prev_winrate_sum = 0.
        for i in range(self.epochs):
            self.maze.reset()
            transition_list = []
            # print("Epoch:%d" %(i))
            origin_token_pos = self.maze.get_token_pos()
            origin_goal = self.maze.get_goal_pos()
            # print("token_pos:", self.maze.token_pos)
            # print("goal:", self.maze.goal)

            #Normal DQN
            for j in range(self.num_moves_limit):
                s = self.maze.get_state()
                if random.random() <= self.epsilon:
                    dir = np.random.randint(0, 3)
                else:
                    dir = self.get_best_action(s)

                s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                transition = [s, dir, r, s_next, is_terminate]
                self.experience_db.add_data(transition)
                transition_list.append(transition)
                if is_terminate or self.maze.get_reward_sum() < self.maze.get_reward_lower_bound():
                    break

            # print("transition_list:")
            # print(transition_list)
            # a = [transition_list[0][0][0][0],transition_list[0][0][0][1]]
            # print(a)
            # print(np.shape(a))
            # Hindsight experience replay
            for j in range(len(transition_list)-1):
                pos = [transition_list[j][0][0][0],transition_list[j][0][0][1]]  #get token_pos
                num = min(len(transition_list)-j-1,hindsight_batch_size)
                # print("----next trans----")
                for k in np.random.choice(len(transition_list)-j-1, num, replace=False):
                    self.maze.reset()
                    self.maze.set_token_pos(pos)
                    # print("virtual token_pos:", self.maze.token_pos)

                    virtual_goal = [transition_list[j+k+1][0][0][0],transition_list[j+k+1][0][0][1]]   #get goal
                    self.maze.set_goal(virtual_goal)
                    # print("virtual goal_pos:", self.maze.goal)
                    s = self.maze.get_state()

                    dir = transition_list[j][1]   #get dir
                    # print("dir:",dir)

                    s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                    transition = [s, dir, r, s_next, is_terminate]
                    # print("sol transition:", transition)
                    self.experience_db.add_data(transition)

            #Add solution data
            self.maze.reset()
            self.maze.set_token_pos(origin_token_pos)
            self.maze.set_goal(origin_goal)
            pos_list, dir_list = self.maze.get_opt_path2()
            for dir in dir_list:
                s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                transition = [s, dir, r, s_next, is_terminate]
                self.experience_db.add_data(transition)
            if r != 1:
                print("Bug:Can't find path",r)
                print("token_pos:", origin_token_pos)
                print("goal:", origin_goal)

            if i % episode_num == 0:
                for _ in range(optimization_num):
                    inputs, answers = self.experience_db.get_data_by_ddqn(self.batch_size, self.gamma)
                    # history = self.model.fit(inputs, answers, epochs=1, batch_size =self.batch_size, verbose=0)
                    train_loss = gl.get_model().train_on_batch(inputs, answers)
                # Update target model
                self.update_target_model()

            if i % 160 == 0:
                sys.stdout.write("Epochs:%d" % (i))

            # Decay learning_rate
            if i%1600 == 0 and i != 0:
                if winrate_sum <= prev_winrate_sum:
                    self.decay_learning_rate(decay=0.5)
                    print("Decay learning rate to:", K.get_value(gl.get_model().optimizer.lr))
                prev_winrate_sum = winrate_sum
                winrate_sum = 0.

            # if i % self.rounds_to_decay_lr == 0 and i != 0:
            #     self.decay_learning_rate()
            #     print("Decay learning rate to:", K.get_value(gl.get_model().optimizer.lr))

            if i % 160 == 0:
                winrate_sum += self.test(self.rounds_to_test)

            if self.rounds_to_save_model != 0 and i % self.rounds_to_save_model == 0:
                gl.get_model().save(self.saved_model_path)
                gl.get_model().save(self.saved_model_path[:-3] + "_target" + self.saved_model_path[-3:])