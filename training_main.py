from machin.machin.frame.algorithms import MADDPG
from machin.machin.utils.logging import default_logger as logger
from copy import deepcopy


import numpy as np
import sys
sys.path.append('C:/D/SUMO/MARL/multiagentRL/')
from gym_sumo.envs import SUMOEnv
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import wandb
from argparse import ArgumentParser

import time
import os
from tqdm import tqdm
import csv
from gym_sumo.envs.utils import generateFlowFiles
from gym_sumo.envs.utils import plot_scores
from gym_sumo.envs.utils import print_status

np.set_printoptions(precision=3, suppress=True)

generateFlowFiles("Train")

max_episodes = 300
max_steps = 10
# number of agents in env, fixed, do not change
hidden_dim = 64
discrete_act = True

if discrete_act:
    batch_size = 512
    actor_learning_rate = 0.01
    critic_learning_rate = 0.01
else:
    batch_size = 1
    actor_learning_rate = 0.001
    critic_learning_rate = 0.005  
tau = 0.01
gamma = 0.95

config = {
  "hidden_layer_sizes": [hidden_dim, hidden_dim],
  "actor_lr": actor_learning_rate,
  "critic_lr": critic_learning_rate,
  "tau": tau,
  "gamma": gamma,
  "max_steps": max_steps,
  "batch_size": batch_size,
  "hidden_dim": hidden_dim,
  "discrete_act": discrete_act
}

use_wandb = os.environ.get('WANDB_MODE', 'disabled') # can be online, offline, or disabled
wandb.init(
  project=f"pytorch_madddpg_SUMO{'MADDPG_Machin'.lower()}",
  tags=["MADDPG_2", "RL"],
#   mode='disabled',
  mode=use_wandb,
  config=config
)
display = 'DISPLAY' in os.environ
use_gui = False
mode = 'gui' if (use_gui and display) else 'none'

# configurations
env = SUMOEnv(mode=mode, discrete_act=discrete_act)
print(env.action_space)
print(env.observation_space)
observe_dim = env._num_observation
action_num = env._num_actions
agent_num = env.n
env.shared_reward = True



if __name__ == "__main__":
    from gym_sumo.actors.maddpg import MADDPGAgent
    from gym_sumo.actors.heuristic import HeuristicAgent

    agent = MADDPGAgent(env, observe_dim, action_num, agent_num, hidden_dim, 
                         batch_size, actor_learning_rate, critic_learning_rate,
                         tau, gamma)
    # agent = HeuristicAgent(env, observe_dim, action_num, agent_num, hidden_dim, 
    #                      batch_size, actor_learning_rate, critic_learning_rate,
    #                      tau, gamma)

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    scores = []

    trainResultFilePath = f"stat_train_{env.pid}.csv"    
    with open(trainResultFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Car_Flow_Rate','Bike_Flow_Rate','Ped_Flow_Rate','Car_Lane_Width','Bike_Lane_Width','Ped_Lane_Width','Co_Sharing','Total_mean_speed_car','Total_mean_speed_bike','Total_mean_speed_ped','Total_Waiting_car_count','Total_Waiting_bike_count','Total_Waiting_ped_count','Total_unique_car_count','Total_unique_bike_count','Total_unique_ped_count', \
                'Total_occupancy_car_Lane','Total_occupancy_bike_Lane','Total_occupancy_ped_Lane','Collision_count_bike','Collision_count_ped','total_density_bike_lane','total_density_ped_lane','total_density_car_lane','RewardAgent_0', 'RewardAgent_1','RewardAgent_2','Hinderance_bb','Hinderance_bp','Hinderance_pp','LevelOfService'])

        for episode in tqdm(range(max_episodes)):
            total_reward = 0
            terminal = False
            step = 0
            states = env.reset('Train')
            states = agent.prepare_states(states) 

            while not terminal and step <= max_steps:
                step += 1
                old_states = states
                # agent model inference
                actions, action_probs = agent.act(states, episode, explore=True)
                print("Actions:", actions, action_probs[-1])

                states, rewards, terminals, info = env.step(actions)
                states = agent.prepare_states(states)
                total_reward += sum(rewards) / agent_num

                # terminal = any(terminals)
                carFlowRate,bikeFlowRate,pedFlowRate,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,total_mean_speed_car,total_mean_speed_bike,total_mean_speed_ped,total_waiting_car_count,total_waiting_bike_count, total_waiting_ped_count,total_unique_car_count,total_unique_bike_count,total_unique_ped_count, \
                car_occupancy,bike_occupancy,ped_occupancy,collision_count_bike,collision_count_ped,total_density_bike_lane,total_density_ped_lane, total_density_car_lane,Hinderance_bb,Hinderance_bp,Hinderance_pp,levelOfService = env.testAnalysisStats()
        
                rewardAgent_0, rewardAgent_1,rewardAgent_2 = env.rewardAnalysisStats()
                writer.writerow([carFlowRate,bikeFlowRate,pedFlowRate,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,total_mean_speed_car,total_mean_speed_bike,total_mean_speed_ped,total_waiting_car_count,total_waiting_bike_count, total_waiting_ped_count,total_unique_car_count,total_unique_bike_count,\
                    total_unique_ped_count,car_occupancy,bike_occupancy,ped_occupancy,collision_count_bike,collision_count_ped,total_density_bike_lane,total_density_ped_lane,total_density_car_lane,rewardAgent_0, rewardAgent_1,rewardAgent_2,Hinderance_bb,Hinderance_bp,Hinderance_pp,levelOfService])
                
                terminals = [term or step == max_steps for term in terminals]
                agent.store_transition(old_states, action_probs, states, rewards, terminals)
                
            
            agent.process_transitions() # process only after an episode
            # total reward is divided by steps here, since:
            # "Agents are rewarded based on minimum agent distance
            #  to each landmark, penalized for collisions"
            total_reward /= step

            # update, update more if episode is longer, else less
            act_loss, crit_loss = None, None
            if episode > 5 and agent.get_buffer_size()>=batch_size:
                for _ in range(step):
                    act_loss, crit_loss = agent.train()

            # show reward
            smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
            scores.append(smoothed_total_reward)
            logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")
            wandb.log({'Actor loss': act_loss,
                       'Critic loss': crit_loss,
                       "Average reward": smoothed_total_reward})
    agent.save('model/Adaptable_street_rl')
    # plot_scores([scores], ['ou'], save_as='normal.png')

    plt.plot(scores)
    plt.xlabel('episodes')
    plt.ylabel('ave rewards')
    plt.savefig('avgScore.jpg')
    # plt.show()

    # plt.figure()
    # plt.plot(avg_cosharing_hist, label=env.reward_agent_2)
    # plt.xlabel('episodes')
    # plt.ylabel('coshare score')
    # plt.legend(loc='upper right')
    # plt.savefig('avgCosharing.jpg')

    # plt.figure()
    # plt.plot(avg_rewardAgent_2_hist)
    # plt.xlabel('episodes')
    # plt.ylabel('agent2 rewards')
    # plt.savefig('avgRewardAgent2.jpg')

    # plt.figure()
    # plt.scatter(avg_cosharing_hist, avg_rewardAgent_2_hist)
    # plt.xlabel('cosharing score')
    # plt.ylabel('agent2 reward')
    # plt.savefig('avgShareReward.jpg')