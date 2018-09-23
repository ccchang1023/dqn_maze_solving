import random
from maze import DIR
import numpy as np
from keras import backend as K

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
        self.rounds_to_test = params.get("rounds_to_test", 100)
        self.checkpoint_file = params.get('checkpoint_file', "")
        
    
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
            for j in range(self.step_limit):
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
                if is_terminate:
                    break

            if i%50 == 0:
                print("Epoch:%d, step_count:%d, reward_sum:%f, loss:%f" %(i, self.maze.get_step_count(), 
                      self.maze.get_reward_sum(), loss))
            if i%self.rounds_to_test==0:
                self.test(self.rounds_to_test)
            
            
    def test(self, rounds=100):
       win_rate = 0.
       average_reward = 0.
       loss = 0.
       for i in range(rounds):
           self.maze.reset()
           for j in range(self.step_limit):
               s = self.maze.get_state()
               dir = self.get_best_action(s)
               _, r, is_goal, is_terminate = self.maze.move(DIR(dir))
               average_reward += r
               if is_goal:
                   win_rate += 1
               if is_terminate:
                   break
       
       inputs, answers = self.experience_db.get_data(self.batch_size)
       loss = self.model.evaluate(inputs, answers, verbose=0)
       win_rate = (win_rate/rounds)*100
       average_reward /= rounds
       output_str = str("Test Result: Loss:%f  Win_rate:%f   Average_reward:%f" %(loss, win_rate, average_reward))
       print(output_str)
        
        
        
