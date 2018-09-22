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
        self.epochs = params.get('epochs', 100)
        self.step_limit = params.get('step_limit', None)
        self.checkpoint_file = params.get('checkpoint_file', "")
        
    
    def initial_dataset(self, n_rounds):
        for _ in range(n_rounds):
            dir = random.randint(0,3)
            # dir = random.choice(list(DIR))
            s = self.maze.get_state()
            s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
            transition = [s,dir,r,s_next,is_terminate]
            self.experience_db.add(transition)
    
    #retrun the action of max Qvalue(predict by model)
    def get_best_action(self, state):
        return np.argmax(self.model.predict(state)) #Return dir of max Qvalue
        
        
    def train(self):
        loss = 0.
        wincount = 0
        for i in range(self.epochs):
            for _ in range(self.step_limit):
                s = self.maze.get_state()
                # dir = random.randint(0,3)
                dir = self.get_best_action(s)
                s_next, r, is_goal, is_terminate = self.maze.move(DIR(dir))
                transition = [s,dir,r,s_next,is_terminate]
                self.experience_db.add(transition)
                # self.maze.create_img()
                inputs, answers = self.experience_db.get_data(self.batch_size, self.gamma)
                history = self.model.fit(inputs, answers, epochs=8, batch_size=16, verbose=0)
                loss = self.model.evaluate(inputs, answers, verbose=0)
                if is_terminate:
                    if is_goal:
                        wincount += 1
                    break
            print("Epoch:%d, step_count:%d, wincount:%d, loss:%f" %(i, self.maze.step_count, wincount, loss) )
            wincount = 0
            self.maze.reset()
            
            
