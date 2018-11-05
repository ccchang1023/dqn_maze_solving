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
        if self.tensorboard_log_path != "":
            if not os.path.isfile(self.tensorboard_log_path):
                str = "mkdir " + self.tensorboard_log_path
                os.system(str)

        step = 0
        for i in range(self.epochs):
            self.maze.reset()
            # print("Epoch:%d" %(i))
            for j in range(self.num_moves_limit):
                s = self.maze.get_state()
                if random.random() <= self.epsilon:
                    dir = np.random.randint(0, 3)
                else:
                    dir = self.get_best_action(s)
                s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                transition = [s, dir, r, s_next, is_terminate]
                self.experience_db.add_data(transition)
                inputs, answers = self.experience_db.get_data_by_ddqn(self.batch_size, self.gamma)
                # history = self.model.fit(inputs, answers, epochs=1, batch_size =self.batch_size, verbose=0)
                train_loss = gl.get_model().train_on_batch(inputs, answers)

                if is_terminate or self.maze.get_reward_sum() < self.maze.get_reward_lower_bound():
                    break

                #Update target model
                step += 1
                if step%self.step_to_update_tModel == 0:
                    step = 0
                    self.update_target_model()


            # Decay learning_rate
            if i % self.rounds_to_decay_lr == 0 and i != 0:
                self.decay_learning_rate()
                print("Decay learning rate to:", K.get_value(gl.get_model().optimizer.lr))

            if i % 100 == 0:
                sys.stdout.write("Epochs:%d" % (i))
                self.test(self.rounds_to_test)

            if self.rounds_to_save_model != 0 and i % self.rounds_to_save_model == 0:
                gl.get_model().save(self.saved_model_path)
                gl.get_model().save(self.saved_model_path[:-3] + "_target" + self.saved_model_path[-3:])
