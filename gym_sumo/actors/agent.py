class Agent:
    def __init__(self, env, observe_dim, action_num, agent_num,
                 hidden_dim, batch_size, actor_learning_rate, critic_learning_rate,
                 tau, gamma):
        self.env = env
        self.observe_dim = observe_dim
        self.action_num = action_num
        self.agent_num = agent_num
        self.tmp_observations_list = [[] for _ in range(agent_num)]


    def act(self, states, epnum, explore=True):
        raise NotImplementedError

    def prepare_states(self, states):
        return states
    
    def train(self):
        raise NotImplementedError
    
    def store_transition(self, old_states, action_probs, states, rewards, terminals):
        raise NotImplementedError

    def process_transitions(self):
        raise NotImplementedError

    def get_buffer_size(self):
        raise NotImplementedError