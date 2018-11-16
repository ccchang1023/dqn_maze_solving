from dqn import DQN
from ddqn import DDQN
import global_setting as gl
from model import dueldqn_model

class DUELDQN(DDQN):
    def __init__(self, **train_params):
        super(DUELDQN, self).__init__(**train_params)
        gl.set_model(dueldqn_model(self.learning_rate, self.maze.get_state().size, self.maze.get_num_of_actions()))
