class DQN(object):        
    def __init__(self, model, maze, **opt):
        self.model = MODEL(None)   
        global epsilon
        n_epoch = opt.get('n_epoch', 15000)
        max_memory = opt.get('max_memory', 1000)
        data_size = opt.get('data_size', 50)
        weights_file = opt.get('weights_file', "")
        name = opt.get('name', 'model')
        start_time = datetime.datetime.now()
    def get_action(self, state):
    
    def train(self,model, maze, **opt):
        for epoch in range(n_epoch):
            loss = 0.0
            rat_cell = random.choice(qmaze.free_cells)
            qmaze.reset(rat_cell)
            game_over = False
            # get initial envstate (1d flattened canvas)
            envstate = qmaze.observe()          
            #transitions = [state, action, reward, next_state, is_terminate]
