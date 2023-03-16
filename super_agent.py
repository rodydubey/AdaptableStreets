import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers as opt
import random
import time
import json
import os
import sys
from config import *
from replay_buffer import *
from agent import *

class SuperAgent:
    def __init__(self, env, path_save=PATH_SAVE_MODEL, path_load=PATH_LOAD_FOLDER):
        self.path_save = path_save
        self.path_load = path_load
        self.replay_buffer = ReplayBuffer(env)
        self.n_agents = len(env.agents)
        self.agents = [Agent(env, agent) for agent in range(self.n_agents)]
        
    def get_actions(self, agents_states,epsilon,evaluation):
        list_actions = [self.agents[index].get_actions(agents_states[index],epsilon,evaluation) for index in range(self.n_agents)]
        # list_actions = []
        # for index in range(self.n_agents):
        #     states = agents_states[index]
        #     if index == 1:
        #         states = agents_states[index] + [list_actions[-1]]
        #     act = self.agents[index].get_actions(states,epsilon,evaluation)
        #     list_actions.append(act)
        return list_actions
    
    def save(self):
        date_now = time.strftime("%Y%m%d%H%M")
        full_path = f"{self.path_save}/save_agent_{date_now}"
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        
        for agent in self.agents:
            agent.save(full_path)
            
        self.replay_buffer.save(full_path)
    
    def load(self):
        full_path = self.path_load
        for agent in self.agents:
            agent.load(full_path)
            
        self.replay_buffer.load(full_path)

    # def printCriticValues(self,states,actions):
    #     aaa = [a.reshape(1,1) for a in actions]
    #     aa = tf.concat(aaa,axis=1)
    #     ss = tf.convert_to_tensor(states)
    #     print(self.agents[0].target_critic(tf.reshape(ss,(1,-1)),tf.reshape(aa,(1,-1))))
    #     print(self.agents[1].target_critic(tf.reshape(ss,(1,-1)),tf.reshape(aa,(1,-1))))
    #     print(self.agents[2].target_critic(tf.reshape(ss,(1,-1)),tf.reshape(aa,(1,-1))))


    
    def train(self):
        if self.replay_buffer.check_buffer_size() == False:
            return
        
        state, reward, next_state, done, actors_state, actors_next_state, actors_action = self.replay_buffer.get_minibatch()
        
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)
        
        actors_states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_state]
        actors_next_states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_next_state]
        actors_actions = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_action]
        
        with tf.GradientTape(persistent=True) as tape:
            target_actions = [self.agents[index].target_actor(actors_next_states[index]) for index in range(self.n_agents)]
            policy_actions = [self.agents[index].actor(actors_states[index]) for index in range(self.n_agents)]
            
            concat_target_actions = tf.concat(target_actions, axis=1)
            concat_policy_actions = tf.concat(policy_actions, axis=1)
            concat_actors_action = tf.concat(actors_actions, axis=1)
            
            target_critic_values = [tf.squeeze(self.agents[index].target_critic(next_states, concat_target_actions), 1) for index in range(self.n_agents)]
            # print(target_critic_values)
            critic_values = [tf.squeeze(self.agents[index].critic(states, concat_actors_action), 1) for index in range(self.n_agents)]
            targets = [rewards[:, index] + self.agents[index].gamma * target_critic_values[index] * (1-done[:, index]) for index in range(self.n_agents)]
            critic_losses = [tf.keras.losses.MSE(targets[index], critic_values[index]) for index in range(self.n_agents)]
            
            actor_losses = [-self.agents[index].critic(states, concat_policy_actions) for index in range(self.n_agents)]
            actor_losses = [tf.math.reduce_mean(actor_losses[index]) for index in range(self.n_agents)]
        
        critic_gradients = [tape.gradient(critic_losses[index], self.agents[index].critic.trainable_variables) for index in range(self.n_agents)]
        actor_gradients = [tape.gradient(actor_losses[index], self.agents[index].actor.trainable_variables) for index in range(self.n_agents)]
        
        for index in range(self.n_agents):
            self.agents[index].critic.optimizer.apply_gradients(zip(critic_gradients[index], self.agents[index].critic.trainable_variables))
            self.agents[index].actor.optimizer.apply_gradients(zip(actor_gradients[index], self.agents[index].actor.trainable_variables))
            self.agents[index].update_target_networks(self.agents[index].tau)