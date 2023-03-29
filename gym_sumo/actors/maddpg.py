import numpy as np
from copy import deepcopy
from machin.machin.frame.algorithms import MADDPG
import torch as t
import torch.nn as nn
from gym_sumo.actors.agent import Agent

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, discrete=True):
        super().__init__()
        # self.in_func = nn.BatchNorm1d(state_dim, momentum=None)
        # self.in_func.weight.data.fill_(1)
        # self.in_func.bias.data.fill_(0)
        self.in_func = lambda x: x
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        if discrete:
            self.output_nonlin = nn.Softmax(dim=1)
        else:
            self.output_nonlin = nn.Sigmoid()

    def forward(self, state):
        a = t.relu(self.fc1(self.in_func(state)))
        a = t.relu(self.fc2(a))
        a = self.output_nonlin(self.fc3(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        # This critic implementation is shared by the prey(DDPG) and
        # predators(MADDPG)
        # Note: For MADDPG
        #       state_dim is the dimension of all states from all agents.
        #       action_dim is the dimension of all actions from all agents.
        super().__init__()
        # self.in_func = nn.BatchNorm1d(state_dim + action_dim, momentum=None)
        # self.in_func.weight.data.fill_(1)
        # self.in_func.bias.data.fill_(0)
        self.in_func = lambda x: x

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(self.in_func(state_action)))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


def create_actors(observe_dim, action_num, hidden_dim):
    actors = []
    for obs_size, act_size in zip(observe_dim, action_num):
        if act_size == 1:
            discrete = False
        else:
            discrete = True
        actors.append(Actor(obs_size, act_size, hidden_dim, discrete=discrete))
        actors[-1].eval()
    return actors


class MADDPGAgent(Agent):
    def __init__(self, env, observe_dim, action_num, agent_num,
                 hidden_dim, batch_size, actor_learning_rate, critic_learning_rate,
                 tau, gamma):
        self.env = env
        self.observe_dim = observe_dim
        self.action_num = action_num
        self.agent_num = agent_num
        self.tmp_observations_list = [[] for _ in range(agent_num)]

        actors = create_actors(observe_dim, action_num, hidden_dim)
        critic = Critic(sum(observe_dim), sum(action_num), hidden_dim)
        critic.eval()

        self.maddpg = MADDPG(
            actors,
            [deepcopy(actor) for actor in actors],
            [deepcopy(critic) for _ in range(agent_num)],
            [deepcopy(critic) for _ in range(agent_num)],
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            critic_visible_actors=[list(range(agent_num))] * agent_num,
            batch_size=batch_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            update_rate=tau,
            discount=gamma,
            use_jit=False
        )

    def act(self, states, epnum):
        with t.no_grad():

            states = [{'state': sk} for sk in states]
            actions = self.maddpg._act_api_general(states, use_target=False)
            result = []

            decayrate = 100
            decay = np.exp(-epnum/decayrate)
            for i, (action, *others) in enumerate(actions):
                if self.action_num[i] == 1:
                    normal_scalar = 0.2
                    noise_uniform = t.randn(1, device=action.device)*normal_scalar
                    action = action + decay*noise_uniform
                    action = t.clip(action, 0.1, 0.9)
                    result.append((action, action))
                else:
                    batch_size = action.shape[0]
                    dist = t.distributions.Categorical(action)
                    action_disc = dist.sample([batch_size, 1]).view(batch_size, 1)
                    result.append((action_disc, action))

            actions = [r[0][0] for r in result]
            action_probs = [r[1] for r in result]
        return actions, action_probs

    def prepare_states(self, states):
        states = [
            t.tensor(st, dtype=t.float32).view(1, self.observe_dim[i]) for i, st in enumerate(states)
        ]
        return states

    def train(self):
        return self.maddpg.update()
    
    def store_transition(self, old_states, action_probs, states, rewards, terminals):
        for tmp_observations, ost, act, st, rew, term in zip(
                    self.tmp_observations_list,
                    old_states,
                    action_probs,
                    states,
                    rewards,
                    terminals,
                ):
                    tmp_observations.append(
                        {
                            "state": {"state": ost},
                            "action": {"action": act},
                            "next_state": {"state": st},
                            "reward": float(rew),
                            "terminal": term,
                        }
                    )

    def process_transitions(self):
        self.maddpg.store_episodes(self.tmp_observations_list)
        self.tmp_observations_list = [[] for _ in range(self.agent_num)]

    def get_buffer_size(self):
        return self.maddpg.replay_buffers[0].size()