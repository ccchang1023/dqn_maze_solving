import numpy as np
from model import default_model


class ExperienceDB(object):
    def __init__(self, model=None, db_cpacity=1000):
        self.model = model
        self.capacity = db_cpacity
        self.data= list()  #sets of transitions
    
    """
    Return input, answer
    input : np.array(batch_size, nrows*ncols)
    answer: np.array(batch_size, num_of_actions)
    keras.model.fit(input, answer, ...)
    """  
    def get_data(self, batch_size=None, gamma=0.95):
        if len(self.data) == 0:
            return None, None
            
        #transitions definition : [state, action, reward, next_state, is_terminate]
        state_size = self.data[0][0].size #Get 1D state size
        num_of_actions = self.model.output_shape[-1]
        inputs = np.zeros((batch_size, state_size))
        answers = np.zeros((batch_size, num_of_actions))
        # print ("Gamma:", gamma)
        for i,j in enumerate(np.random.choice(len(self.data), batch_size, replace=False)):
            state, action, reward, next_state, is_terminate = self.data[j]
            inputs[i] = state.flatten()  #reshape state to (nrows*ncols)
            
            #**Test this line when run successfully
            answers[i] = self.model.predict(state).flatten() #reshape to (num_of_actions)
            
            if is_terminate:
                answers[i][action] = reward
            else:
                qvalue_next = np.max(self.model.predict(next_state)) #get max qvalue
                answers[i][action] = reward + (gamma*qvalue_next)
            
        return inputs, answers
        

    def add(self, transition=None):
        #transitions = [state, action, reward, next_state, is_terminate]
        self.data.append(transition)
        if len(self.data) > self.capacity:
            self.pop_from_head()
            
    def pop_from_head(self):
        del self.data[0]
        
    def show_data(self):
        for i in self.data:
            print(i, "\n")
    
    def get_data_list(self):
        return self.data
        
        
        
        
        
        