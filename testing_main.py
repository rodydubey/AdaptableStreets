import numpy as np
import sys
sys.path.append('C:/D/SUMO/MARL/multiagentRL/')
from gym_sumo.envs import SUMOEnv
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from argparse import ArgumentParser
from tqdm import tqdm

from config import *
from replay_buffer import *
from networks import *
from agent import *
from super_agent import *
import csv

config = dict(
  learning_rate_actor = ACTOR_LR,
  learning_rate_critic = CRITIC_LR,
  batch_size = BATCH_SIZE,
  architecture = "MADDPG",
  infra = "Colab",
  env = ENV_NAME
)

# env = gym.make('SumoGUI-v0')
env = SUMOEnv(mode='gui')
print(env.action_space)
print(env.observation_space)
super_agent = SuperAgent(env)

scores = []
score_history = []
cumm_test_score_list = []
cumm_queue_length_car = []
cumm_queue_length_bike = []
cumm_queue_length_ped = [] 
epsilon = 0
evaluation = True

if PATH_LOAD_FOLDER is not None:
    print("loading weights")
    actors_state = env.reset(True)
    actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)],epsilon,evaluation)
    [super_agent.agents[index].target_actor(actors_state[index][None, :]) for index in range(super_agent.n_agents)]
    state = np.concatenate(actors_state)
    actors_action = np.concatenate(actors_action)
    [super_agent.agents[index].critic(state[None, :], actors_action[None, :]) for index in range(super_agent.n_agents)]
    [super_agent.agents[index].target_critic(state[None, :], actors_action[None, :]) for index in range(super_agent.n_agents)]
    super_agent.load()

    print(super_agent.replay_buffer.buffer_counter)
    print(super_agent.replay_buffer.n_games)

    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Cum_Car_Queue_Length','Cum_Bike_Queue_Length','Cum_Ped_Queue_Length'])
        score = 0
        # for n_game in tqdm(range(1)):
            # if super_agent.replay_buffer.check_buffer_size():
        done = [False for index in range(super_agent.n_agents)]
        actors_state = env.reset(False)
        step = 0
        # while not any(done): 
        for step in tqdm(range(50)):
            actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)],epsilon,evaluation)
            actors_next_state, reward, done, info = env.step(actors_action)
            state = np.concatenate(actors_state)
            next_state = np.concatenate(actors_next_state)
            actors_state = actors_next_state        
            score += sum(reward)
            cumm_test_score_list.append(score)
            print(score)
            queue_length_car, queue_length_bike, queue_length_ped = env.QueueLength()
            cumm_queue_length_car.append(queue_length_car)
            cumm_queue_length_bike.append(queue_length_bike)
            cumm_queue_length_ped.append(queue_length_ped)
            writer.writerow([queue_length_car,queue_length_bike, queue_length_ped])

                    # step += 1
                    # if step >= MAX_STEPS:
                    #     break
        # plt.plot(cumm_test_score_list)
        # plt.show()
        
        #Plot Cummulative Queue Length for each vehicle type
        plt.plot(cumm_queue_length_car)
        plt.show()

else:
    print("No model loaded")
