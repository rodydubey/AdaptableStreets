from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from gym_sumo.envs.utils import print_status
from gym_sumo.envs.utils import plot_scores
from gym_sumo.envs.utils import generateFlowFiles
import csv
from tqdm import tqdm
from argparse import ArgumentParser
import wandb
import warnings
from matplotlib import pyplot as plt
from gym_sumo.envs import SUMOEnv
import argparse
import gym
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete, Tuple
from pathlib import Path
from torch.autograd import Variable
# from tensorboardX import SummaryWriter

from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG

import numpy as np
import sys
sys.path.append('C:/D/SUMO/MARL/multiagentRL/')
warnings.filterwarnings('ignore')

display = 'DISPLAY' in os.environ
use_gui = False
mode = 'gui' if (use_gui and display) else 'none'

USE_CUDA = False  # torch.cuda.is_available()

# mode = 'gui'
EDGES = ['E0']
joint_agents = False
# EDGES = ['E0','-E1','-E2','-E3']
# joint_agents = True
# generateFlowFiles("Test 0")
env_kwargs = {'mode': mode,
              'edges': EDGES,
              'joint_agents': joint_agents}
# generateFlowFiles("Test 0")
# env = SUMOEnv(**env_kwargs)

from training_ppo import SUMOEnvPPO
env = SUMOEnvPPO(**env_kwargs)

# env = make_vec_env(SUMOEnvPPO, n_envs=1, env_kwargs=env_kwargs)
# check_env(env, warn=True)
# env.env_method('set_run_mode', 'Test')

print(env.action_space)
print(env.action_space.sample())
print(env.observation_space)
# env.reset()
model = PPO("MlpPolicy", env, n_steps=6, verbose=1)


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    # run_id = 'run225'
    run_id = 'ppo_4.87'
    run_dir = model_dir / run_id
    model.load(run_dir / 'model.pt')
    t = 0
    scores = []
    smoothed_total_reward = 0
    start_seed = 42
    num_seeds = 1
    run_mode = 'Test'
    surge = True
    env.set_run_mode('Test', surge=surge)

    # testResultFilePath = f"results/PPO_test_{'surge' if surge else 'nosurge'}_{run_id}.csv"
    testResultFilePath = f"results/debug_ppo.csv"
    with open(testResultFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['Car_Flow_Rate','Bike_Flow_Rate','Ped_Flow_Rate','Car_Lane_Width','Bike_Lane_Width','Ped_Lane_Width','Co_Sharing', \
        #         'Total_occupancy_car_Lane','Total_occupancy_bike_Lane','Total_occupancy_ped_Lane','total_density_bike_lane','total_density_ped_lane','total_density_car_lane','RewardAgent_0', 'RewardAgent_1','RewardAgent_2','LevelOfService'])

        # writer.writerow(['avg_waiting_time_car','avg_waiting_time_bike','avg_waiting_time_ped','avg_queue_length_car','avg_queue_length_bike','avg_queue_length_ped','los',"Reward_Agent_2","cosharing",'timeslot'])
        written_headers = False
        if num_seeds>1:
            seed_list = list(range(start_seed,start_seed+num_seeds))
        else:
            seed_list = [start_seed]
        for seed in seed_list: # realizations for averaging
            env.set_sumo_seed(seed)
            env.timeOfHour = 1 # hack
            # env.modeltype = modeltype # hack
            env.firstTimeFlag = True

            # env.envs[0].set_sumo_seed(seed)
            # env.set_attr('timeOfHour', 1)  # hack
            # env.envs[0].firstTimeFlag = True
            for ep_i in tqdm(range(0, config.n_episodes)):
                total_reward = 0
                print("Episodes %i-%i of %i" % (ep_i + 1,
                                                ep_i + 1 + config.n_rollout_threads,
                                                config.n_episodes), env.timeOfHour)

                obs = env.reset()
                step = 0
                for et_i in range(config.episode_length):
                    print("Step {}".format(step))
                    step += 1
                    action, _ = model.predict(obs, deterministic=True)
                    print("Action: ", action)
                    next_obs, reward, done, info = env.step(action)
                    obs = next_obs
                    total_reward += reward
                    # avg_waiting_time_car,avg_waiting_time_bike,avg_waiting_time_ped,avg_queue_length_car,avg_queue_length_bike,avg_queue_length_ped,los,reward_agent_2,cosharing = env.getTestStats()

                    # # rewardAgent_2 = 0
                    # writer.writerow([avg_waiting_time_car,avg_waiting_time_bike,avg_waiting_time_ped,avg_queue_length_car,avg_queue_length_bike,avg_queue_length_ped,los,reward_agent_2,cosharing,ep_i])
                    for edge_agent in env.edge_agents:
                        headers, values = edge_agent.getTestStats()
                        if not written_headers:
                            writer.writerow(headers + ['timeslot', 'seed'])
                            written_headers = True
                        writer.writerow(values + [ep_i, seed])
                total_reward /= step
                # show reward
                smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
                scores.append(smoothed_total_reward)
                # print('obs=', obs, 'reward=', reward, 'done=', done)
    plt.plot(scores)
    plt.xlabel('episodes')
    plt.ylabel('ave rewards')
    plt.savefig('avgScore.jpg')
    plt.show()
    # model.save(run_dir / 'model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple", type=str)
    parser.add_argument("--model_name", default="simple_model", type=str)
    # parser.add_argument("--run_id", default="run12", type=str) # run47 is performing the best on training data
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=48, type=int)
    parser.add_argument("--episode_length", default=6, type=int)
    parser.add_argument("--steps_per_update", default=10, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=30, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
