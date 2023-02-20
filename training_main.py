import numpy as np
import sys
sys.path.append('C:/D/SUMO/MARL/multiagentRL/')
import gym
import gym_sumo
import random
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import wandb
from argparse import ArgumentParser
from tensorflow import keras
import tensorflow as tf
import time
import json
import os
from tqdm import tqdm

from config import *
from replay_buffer import *
from networks import *
from agent import *
from super_agent import *

config = dict(
  learning_rate_actor = ACTOR_LR,
  learning_rate_critic = CRITIC_LR,
  batch_size = BATCH_SIZE,
  architecture = "MADDPG",
  infra = "Colab",
  env = ENV_NAME
)

wandb.init(
  project=f"tensorflow2_madddpg_SUMO{ENV_NAME.lower()}",
  tags=["MADDPG", "RL"],
  config=config,
)

parser = ArgumentParser()
parser.add_argument('--action',help='"train_from_scratch" or "resume_training", or "test"')
args = parser.parse_args()

env = gym.make('SumoGUI-v0')
print(env.action_space)
print(env.observation_space)
super_agent = SuperAgent(env)

scores = []
score_history = []
avg_score_list = []
PRINT_INTERVAL = 101
epsilon = 0
evaluation = True
if PATH_LOAD_FOLDER is not None:
    print("loading weights")
    actors_state = env.reset(False)
    actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)],epsilon,evaluation)
    [super_agent.agents[index].target_actor(actors_state[index][None, :]) for index in range(super_agent.n_agents)]
    state = np.concatenate(actors_state)
    actors_action = np.concatenate(actors_action)
    [super_agent.agents[index].critic(state[None, :], actors_action[None, :]) for index in range(super_agent.n_agents)]
    [super_agent.agents[index].target_critic(state[None, :], actors_action[None, :]) for index in range(super_agent.n_agents)]
    super_agent.load()

    print(super_agent.replay_buffer.buffer_counter)
    print(super_agent.replay_buffer.n_games)

for n_game in tqdm(range(MAX_GAMES)):
    start_time = time.time()
    actors_state = env.reset(False)
    done = [False for index in range(super_agent.n_agents)]
    score = 0
    step = 0
    evaluation = False
    # print("start")
    epsilon = 1.0 - (n_game / MAX_GAMES)
    # if n_game > MAX_GAMES - 200:
    #     if epsilon > 0.1:
    #         epsilon = 0.1
    # loop through 20 times 600 simulation steps
    while not any(done):      
        # print("Epsilon :" + str(epsilon))
        actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)],epsilon,evaluation)
        
        actors_next_state, reward, done, info = env.step(actors_action)
        if step >= MAX_STEPS:
            done = True
        state = np.concatenate(actors_state)
        next_state = np.concatenate(actors_next_state)
        
        super_agent.replay_buffer.add_record(actors_state, actors_next_state, actors_action, state, next_state, reward, done)
        
        actors_state = actors_next_state
        
        score += sum(reward) 
        score_history.append(score)
        step += 1
        if step >= MAX_STEPS:
            break
    
    
    # if super_agent.replay_buffer.check_buffer_size():
    #     super_agent.train()
    if n_game % TRAINING_STEP == 0 and n_game > 0: 
        super_agent.train()
        print("Training Episode: " + str(n_game))
        
    super_agent.replay_buffer.update_n_games()
    
    scores.append(score)
    # average of last 100 scores
    avg_score = np.mean(score_history[-100:])
    avg_score_list.append(avg_score)
    if n_game % PRINT_INTERVAL == 0 and n_game > 0:
        print('episode', n_game, 'average score {:.1f}'.format(avg_score))

    
    wandb.log({'Game number': super_agent.replay_buffer.n_games, '# Episodes': super_agent.replay_buffer.buffer_counter, 
                "Average reward": round(np.mean(scores[-10:]), 2), \
                      "Time taken": round(time.time() - start_time, 2)})
    
    if (n_game+1) % EVALUATION_FREQUENCY == 0 and super_agent.replay_buffer.check_buffer_size():
        actors_state = env.reset(False)
        done = [False for index in range(super_agent.n_agents)]
        score = 0
        step = 0        
        evaluation = True
        while not any(done):
            actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)],epsilon,evaluation)
            actors_next_state, reward, done, info = env.step(actors_action)
            state = np.concatenate(actors_state)
            next_state = np.concatenate(actors_next_state)
            actors_state = actors_next_state
            score += sum(reward)
            step += 1
            if step >= MAX_STEPS:
                break
        wandb.log({'Game number': super_agent.replay_buffer.n_games, 
                   '# Episodes': super_agent.replay_buffer.buffer_counter, 
                   'Evaluation score': score})
            
    if (n_game + 1) % SAVE_FREQUENCY == 0:
        print("saving weights and replay buffer...")
        super_agent.save()
        print("saved")


plt.plot(avg_score_list)
plt.savefig('results/avgScore.jpg')
plt.show()