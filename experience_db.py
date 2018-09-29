import random
import numpy as np
from model import default_model


class ExperienceDB(object):
    def __init__(self, model=None, db_cpacity=1000):
        self.model = model
        self.capacity = db_cpacity
        self.data= list()  #list shape: [data_size, 5]
        self.game_order_data_set = set() #set shape : [data_size, step sum in a game, 5]
        
    """
    Return input, answer
    input : np.array(batch_size, nrows*ncols)
    answer: np.array(batch_size, num_of_actions)
    keras.model.fit(input, answer, ...)
    """  
    def get_data(self, batch_size=None, gamma=0.95, goal_data_ratio=0.5):
        if len(self.data) == 0:
            return None, None
        #transitions definition : [state, action, reward, next_state, is_terminate]
        state_size = self.data[0][0].size #Get 1D state size
        num_of_actions = self.model.output_shape[-1]
        inputs = np.zeros((batch_size, state_size))
        answers = np.zeros((batch_size, num_of_actions))
        # print ("Gamma:", gamma)
        for i,j in enumerate(np.random.choice(len(self.data), batch_size, replace=False)):
            state, dir, reward, next_state, is_terminate = self.data[j]
            inputs[i] = state.flatten()  #reshape state to (nrows*ncols)
            
            answers[i] = self.model.predict(state).flatten() #reshape to (num_of_actions)
            
            if is_terminate:
                answers[i][dir] = reward
            else:
                qvalue_next = np.max(self.model.predict(next_state)) #get max qvalue
                answers[i][dir] = reward + (gamma*qvalue_next)
        
        #Collect goal data
        goal_state_list = list()
        goal_answer_list = list()
        sample_num = int(batch_size*goal_data_ratio)
        if sample_num > len(self.game_order_data_set):
            sample_num = len(self.game_order_data_set)
        count = 0
        
        if len(self.game_order_data_set) > 0:
            transition = random.sample(self.game_order_data_set , 1)
            print("transition shape:", np.shape(transition))
            for state, dir, reward, next_state, is_terminate in transition:
                goal_state_list.append(state.flatten())
                action_value = self.model.predict(next_state).flatten()
                if is_terminate:
                    action_value[dir] = reward
                else:
                    qvalue_next = np.max(self.model.predict(next_state))
                    action_value[dir] = reward + (gamma*qvalue_next)
                goal_answer_list.append(action_value)
        
        # print("goal_state_list shape:", np.shape(goal_state_list))
        # print("goal_answer_list shape:", np.shape(goal_answer_list))

        if len(goal_state_list) != 0:
            inputs = np.append(inputs, goal_state_list, axis=0)
            answers = np.append(answers, goal_answer_list, axis=0)
            
            # print("shape of input:", np.shape(inputs))
            # print("shape of answers:", np.shape(answers))
        
        #For Conv2D input
        # _, r, c, _ = state.shape
        # inputs = inputs.reshape(batch_size,r,c,1)
       
        return inputs, answers
        
 
    
    def add_data(self, transition=None):
        #transitions = [state, action, reward, next_state, is_terminate]
        self.data.append(transition)
        if len(self.data) > self.capacity:
            self.pop_data()
    
    def add_game_order_data(self, transition_list=None):
        t = tuple(transition_list)
        self.game_order_data_set.add(t) #Only tuple can be hashable
        print("Current order data len:", len(self.game_order_data_set))
        if(len(self.game_order_data_set) > 100):
           self.del_order_data_from_set()
    
    def pop_data(self):
        del self.data[0]
    
    def del_order_data_from_set(self):
        self.game_order_data_set.pop()

    
    def show_data(self):
        for i in self.data:
            print(i, "\n")
    
    def get_data_list(self):
        return self.data
        
        
        
        
        
        