from machin.machin.frame.algorithms import MADDPG
from machin.machin.utils.logging import default_logger as logger
from copy import deepcopy
import torch as t
import torch.nn as nn

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

# Important note:
# In order to successfully run the environment, please git clone the project
# then run:
#    pip install -e ./test_lib/multiagent-particle-envs/
# in project root directory
use_wandb = os.environ.get('WANDB_MODE', 'disabled') # can be online, offline, or disabled
wandb.init(
  project=f"pytorch_madddpg_SUMO{'MADDPG_Machin'.lower()}",
  tags=["MADDPG_2", "RL"],
  mode=use_wandb
)
display = 'DISPLAY' in os.environ
use_gui = False
mode = 'gui' if (use_gui and display) else 'none'

# configurations
env = SUMOEnv(mode=mode)
print(env.action_space)
print(env.observation_space)
# env.discrete_action_input = True
observe_dim = env._num_observation
action_num = env._num_actions
max_episodes = 300
max_steps = 5
# number of agents in env, fixed, do not change
agent_num = 3

# model definition
class ActorDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.softmax(self.fc3(a), dim=1)
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        # This critic implementation is shared by the prey(DDPG) and
        # predators(MADDPG)
        # Note: For MADDPG
        #       state_dim is the dimension of all states from all agents.
        #       action_dim is the dimension of all actions from all agents.
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


if __name__ == "__main__":
    actors = [ActorDiscrete(obs_size, act_size) for obs_size, act_size in zip(observe_dim, action_num)]
    critic = Critic(sum(observe_dim), sum(action_num))

    maddpg = MADDPG(
        actors,
        [deepcopy(actor) for actor in actors],
        [deepcopy(critic) for _ in range(agent_num)],
        [deepcopy(critic) for _ in range(agent_num)],
        t.optim.Adam,
        nn.MSELoss(reduction="sum"),
        critic_visible_actors=[list(range(agent_num))] * agent_num,
        batch_size = 8
    )

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    scores = []
    trainResultFilePath = f"stat_train_{env.pid}.csv"    
    with open(trainResultFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Car_Flow_Rate','Bike_Flow_Rate','Ped_Flow_Rate','Car_Lane_Width','Bike_Lane_Width','Ped_Lane_Width','Co_Sharing','Total_mean_speed_car','Total_mean_speed_bike','Total_mean_speed_ped','Total_Waiting_car_count','Total_Waiting_bike_count','Total_Waiting_ped_count','Total_unique_car_count','Total_unique_bike_count','Total_unique_ped_count', \
                'Total_occupancy_car_Lane','Total_occupancy_bike_Lane','Total_occupancy_ped_Lane','Collision_count_bike','Collision_count_ped','total_density_bike_lane','total_density_ped_lane','total_density_car_lane','RewardAgent_0', 'RewardAgent_1','RewardAgent_2','Hinderance_bb','Hinderance_bp','Hinderance_pp','LevelOfService'])

        while episode < max_episodes:
            episode += 1
            total_reward = 0
            terminal = False
            step = 0
            states = [
                t.tensor(st, dtype=t.float32).view(1, observe_dim[i]) for i, st in enumerate(env.reset("Train"))
            ]
            tmp_observations_list = [[] for _ in range(agent_num)]

            while not terminal and step <= max_steps:
                step += 1
                with t.no_grad():
                    old_states = states
                    # agent model inference
                    results = maddpg.act_discrete_with_noise(
                        [{"state": st} for st in states]
                    )
                    actions = [int(r[0]) for r in results]
                    action_probs = [r[1] for r in results]
                    print("before step")
                    print(actions)
                    states, rewards, terminals, info = env.step(actions)
                    states = [
                        t.tensor(st, dtype=t.float32).view(1, observe_dim[i]) for i, st in enumerate(states)
                    ]
                    total_reward += float(sum(rewards)) / agent_num


                    carFlowRate,bikeFlowRate,pedFlowRate,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,total_mean_speed_car,total_mean_speed_bike,total_mean_speed_ped,total_waiting_car_count,total_waiting_bike_count, total_waiting_ped_count,total_unique_car_count,total_unique_bike_count,total_unique_ped_count, \
                    car_occupancy,bike_occupancy,ped_occupancy,collision_count_bike,collision_count_ped,total_density_bike_lane,total_density_ped_lane, total_density_car_lane,Hinderance_bb,Hinderance_bp,Hinderance_pp,levelOfService = env.testAnalysisStats()
            
                    rewardAgent_0, rewardAgent_1,rewardAgent_2 = env.rewardAnalysisStats()
                    writer.writerow([carFlowRate,bikeFlowRate,pedFlowRate,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,total_mean_speed_car,total_mean_speed_bike,total_mean_speed_ped,total_waiting_car_count,total_waiting_bike_count, total_waiting_ped_count,total_unique_car_count,total_unique_bike_count,\
                        total_unique_ped_count,car_occupancy,bike_occupancy,ped_occupancy,collision_count_bike,collision_count_ped,total_density_bike_lane,total_density_ped_lane,total_density_car_lane,rewardAgent_0, rewardAgent_1,rewardAgent_2,Hinderance_bb,Hinderance_bp,Hinderance_pp,levelOfService])
                    

                    for tmp_observations, ost, act, st, rew, term in zip(
                        tmp_observations_list,
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
                                "terminal": term or step == max_steps,
                            }
                        )
            wandb.log({'# Episodes': episode, 
                "Average reward": round(np.mean(scores[-10:]), 2)})
            
            maddpg.store_episodes(tmp_observations_list)
            # total reward is divided by steps here, since:
            # "Agents are rewarded based on minimum agent distance
            #  to each landmark, penalized for collisions"
            total_reward /= step

            # update, update more if episode is longer, else less
            if episode > 10:
                for _ in range(step):
                    maddpg.update()

            # show reward
            smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
            scores.append(smoothed_total_reward)
            logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        
    
    plot_scores([scores], ['ou'], save_as='normal.png')

    plt.plot(scores)
    plt.xlabel('episodes')
    plt.ylabel('ave rewards')
    plt.savefig('avgScore.jpg')
    plt.show()

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