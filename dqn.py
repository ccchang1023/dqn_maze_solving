import random
from maze import DIR
import numpy as np

class DQN(object):        
    def __init__(self, maze=None, model=None, experience_db=None, **params):
        self.maze = maze
        self.model = model
        self.experience_db = experience_db
        self.batch_size = params.get('batch_size', 10)
        self.gamma = params.get('gamma', 0.95)
        self.epsilon = params.get('epsilon', 0.)
        self.n_epoch = params.get('n_epoch', 10)
        self.checkpoint_file = params.get('checkpoint_file', "")
        
    
    def initial_dataset(self, n_rounds):
        for _ in range(n_rounds):
            dir = random.randint(0,3)
            # dir = random.choice(list(DIR))
            s = self.maze.get_state()
            s_next, r, is_valid, is_terminate = self.maze.move(DIR(str(dir)))
            transition = [s,dir,r,s_next,is_terminate]
            # print(np.shape(transition))
            self.experience_db.add(transition)
            # self.maze.create_img()
    
    def get_action(self, state):
        return
        
    def train(self):
        
        while True:
            for i in range(10):
                dir = random.randint(0,3)
                s = self.maze.get_state()
                s_next, r, is_valid, is_terminate = self.maze.move(DIR(str(dir)))
                transition = [s,dir,r,s_next,is_terminate]
                self.experience_db.add(transition)
                # self.maze.create_img()
            
            inputs, answers = self.experience_db.get_data(self.batch_size, self.gamma)
            print(np.shape(inputs))
            print(np.shape(answers))
            self.maze.reset()
        return
        
        for epoch in range(n_epoch):
            loss = 0.0
            rat_cell = random.choice(qmaze.free_cells)
            qmaze.reset(rat_cell)
            game_over = False
            # get initial envstate (1d flattened canvas)
            envstate = qmaze.observe()          
            #transitions = [state, action, reward, next_state, is_terminate]
