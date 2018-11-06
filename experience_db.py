import random
import numpy as np
import global_setting as gl
from collections import deque


class ExperienceDB(object):
    def __init__(self, db_cpacity=1000, state_size=0):
        self.capacity = db_cpacity
        self.data= deque()  #queue shape: [data_size, 5]
        self.solution_data = deque()
        self.num_of_actions = gl.get_model().output_shape[-1]
        self.maze_state_size = state_size

    """
    Return input, answer
    input : np.array(batch_size, nrows*ncols)
    answer: np.array(batch_size, num_of_actions)
    keras.model.fit(input, answer, ...)
    """  
    def get_data(self, batch_size=None, gamma=0.95, solution_data_ratio=0.3):

        if len(self.data) == 0:
            return None, None
        #transitions definition : [state, action, reward, next_state, is_terminate]
        inputs = np.zeros((batch_size, self.maze_state_size))
        answers = np.zeros((batch_size, self.num_of_actions))
        # print ("Gamma:", gamma)
        sol_num = int(batch_size*solution_data_ratio)

        #Get from solution db
        for i, j in enumerate(np.random.choice(len(self.solution_data), sol_num, replace=False)):
            state, dir, reward, next_state, is_terminate = self.solution_data[j]
            inputs[i] = state.flatten()  # reshape state to (nrows*ncols)
            answers[i] = gl.get_model().predict(state).flatten()  # reshape to (num_of_actions)
            if is_terminate:
                answers[i][dir] = reward
            else:
                qvalue_next = np.max(gl.get_model().predict(next_state))  # get max qvalue
                answers[i][dir] = reward + (gamma * qvalue_next)

        for i,j in enumerate(np.random.choice(len(self.data), batch_size-sol_num, replace=False)):
            state, dir, reward, next_state, is_terminate = self.data[j]
            inputs[i+sol_num] = state.flatten()  #reshape state to (nrows*ncols)
            answers[i+sol_num] = gl.get_model().predict(state).flatten() #reshape to (num_of_actions)
            if is_terminate:
                answers[i+sol_num][dir] = reward
            else:
                qvalue_next = np.max(gl.get_model().predict(next_state)) #get max qvalue
                answers[i+sol_num][dir] = reward + (gamma*qvalue_next)

        #For Conv2D input
        # _, r, c, _ = state.shape
        # inputs = inputs.reshape(batch_size,r,c,1)

        return inputs, answers

    def get_data_by_ddqn(self, batch_size=None, gamma=0.95):
        if len(self.data) == 0:
            return None, None
        # transitions definition : [state, action, reward, next_state, is_terminate]
        inputs = np.zeros((batch_size, self.maze_state_size))
        answers = np.zeros((batch_size, self.num_of_actions))
        # print ("Gamma:", gamma)
        for i, j in enumerate(np.random.choice(len(self.data), batch_size, replace=False)):
            state, dir, reward, next_state, is_terminate = self.data[j]
            inputs[i] = state.flatten()
            answers[i] = gl.get_model().predict(state).flatten()  # reshape to (num_of_actions)

            if is_terminate:
                answers[i][dir] = reward
            else:
                max_action_by_main_model = np.argmax(gl.get_model().predict(next_state))
                # print(max_action_by_main_model)
                qnext_by_target_model = gl.get_targetModel().predict(next_state).squeeze(axis=0)
                # print("and ", qnext_by_target_model)
                qvalue_next = qnext_by_target_model[max_action_by_main_model]
                answers[i][dir] = reward + (gamma * qvalue_next)

        return inputs, answers


    def add_data(self, transition=None):
        #transitions = [state, action, reward, next_state, is_terminate]
        self.data.append(transition)
        if len(self.data) > self.capacity:
            self.data.popleft()

    def add_solution_data(self, transition=None):
        self.solution_data.append(transition)
        if len(self.solution_data) > self.capacity:
            self.solution_data.popleft()

    def pop_data_from_head(self):
        del self.data[0]

    def show_data(self):
        for i in self.data:
            print(i, "\n")
    
    def get_data_list(self):
        return self.data
        
        
        
        
        
        