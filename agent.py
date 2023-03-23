import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers as opt
import time
import json
import os
import sys
import copy
from config import *
from replay_buffer import *
from networks import *
from utils import get_space_dims

np.random.seed(42)
THETA=0.15
DT=1e-1
def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class Agent:
    def __init__(self, env, n_agent, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, 
                 gamma=GAMMA, tau=TAU, noise_sigma=0.15):
        
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.noise_sigma = noise_sigma

        self.actor_dims = get_space_dims(env.observation_space[n_agent])
        self.n_actions = get_space_dims(env.action_space[n_agent])
        
        self.id = n_agent
        self.name = f'agent {self.id}'
        self.agent_name = "agent_number_{}".format(n_agent)
        
        self.actor = Actor("actor_" + self.agent_name, self.n_actions)
        self.critic = Critic("critic_" + self.agent_name)
        self.target_actor = Actor("target_actor_" + self.agent_name, self.n_actions)
        self.target_critic = Critic("critic_" + self.agent_name)
        
        self.actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        
        self.target_actor.set_weights(actor_weights)
        self.target_critic.set_weights(critic_weights)
        
    def update_target_networks(self, tau):
        actor_weights = self.actor.weights
        target_actor_weights = self.target_actor.weights
        for index in range(len(actor_weights)):
            target_actor_weights[index] = tau * actor_weights[index] + (1 - tau) * target_actor_weights[index]

        self.target_actor.set_weights(target_actor_weights)
        
        critic_weights = self.critic.weights
        target_critic_weights = self.target_critic.weights
    
        for index in range(len(critic_weights)):
            target_critic_weights[index] = tau * critic_weights[index] + (1 - tau) * target_critic_weights[index]

        self.target_critic.set_weights(target_critic_weights)

    
    def get_actions(self, actor_states,epsilon,evaluation=False):
        # noise = tf.random.uniform(shape=[self.n_actions])
        # ou_noise = OUNoise(1)
        # noise = ou_noise.sample()
        normal_scalar = self.noise_sigma
        noise_uniform = np.random.randn(self.n_actions) * normal_scalar
        actions = self.actor(actor_states)
        # print(actions)
        if not evaluation:
            if np.random.random() < epsilon: # Decide whether to perform an explorative or exploitative action, according to an epsilon-greedy policy during non-evaluation phase
                actions = actions + noise_uniform
            else:
                print("exploitation")
        # ou_noise.reset() 
        # actions = actions + noise_uniform         
        actions = np.clip(actions.numpy()[0],0.1,0.9)
        print(self.agent_name,actions)
        return actions
    
    def save(self, path_save):
        self.actor.save_weights(f"{path_save}/{self.actor.net_name}.h5")
        self.target_actor.save_weights(f"{path_save}/{self.target_actor.net_name}.h5")
        self.critic.save_weights(f"{path_save}/{self.critic.net_name}.h5")
        self.target_critic.save_weights(f"{path_save}/{self.target_critic.net_name}.h5")
        
    def load(self, path_load):
        self.actor.load_weights(f"{path_load}/{self.actor.net_name}.h5")
        self.target_actor.load_weights(f"{path_load}/{self.target_actor.net_name}.h5")
        self.critic.load_weights(f"{path_load}/{self.critic.net_name}.h5")
        self.target_critic.load_weights(f"{path_load}/{self.target_critic.net_name}.h5")