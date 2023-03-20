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
from config import *
from replay_buffer import *
from networks import *
from agent import *
from super_agent import *
from gym_sumo.envs.utils import generateFlowFiles
from gym_sumo.envs.utils import plot_scores
from gym_sumo.envs.utils import print_status

config = dict(
  learning_rate_actor = ACTOR_LR,
  learning_rate_critic = CRITIC_LR,
  batch_size = BATCH_SIZE,
  architecture = "MADDPG",
  infra = "Colab",
  env = ENV_NAME
)

use_wandb = os.environ.get('WANDB_MODE', 'disabled') # can be online, offline, or disabled
wandb.init(
  project=f"tensorflow2_madddpg_SUMO{ENV_NAME.lower()}",
  tags=["MADDPG", "RL"],
  config=config,
  mode=use_wandb
)

parser = ArgumentParser()
parser.add_argument('--action',help='"train_from_scratch" or "resume_training", or "test"')
args = parser.parse_args()

display = 'DISPLAY' in os.environ
use_gui = False
mode = 'gui' if (use_gui and display) else 'none'
env = SUMOEnv(mode=mode)
print(env.action_space)
print(env.observation_space)
super_agent = SuperAgent(env)

scores = []
score_history = []
avg_score_list = []
PRINT_INTERVAL = 101
epsilon = 0
evaluation = True

#generate training files 
generateFlowFiles("Train")
if PATH_LOAD_FOLDER is not None:
    print("Edit configuration file")
    exit()


trainResultFilePath = f"stat_train_{env.pid}.csv"    
with open(trainResultFilePath, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Car_Flow_Rate','Bike_Flow_Rate','Ped_Flow_Rate','Car_Lane_Width','Bike_Lane_Width','Ped_Lane_Width','Co_Sharing','Total_mean_speed_car','Total_mean_speed_bike','Total_mean_speed_ped','Total_Waiting_car_count','Total_Waiting_bike_count','Total_Waiting_ped_count','Total_unique_car_count','Total_unique_bike_count','Total_unique_ped_count', \
             'Total_occupancy_car_Lane','Total_occupancy_bike_Lane','Total_occupancy_ped_Lane','Collision_count_bike','Collision_count_ped','total_density_bike_lane','total_density_ped_lane','total_density_car_lane','RewardAgent_0', 'RewardAgent_1','RewardAgent_2','Hinderance_bb','Hinderance_bp','Hinderance_pp','LevelOfService'])

    for n_game in tqdm(range(MAX_GAMES)):
        start_time = time.time()
        actors_state = env.reset("Train")
        done = [False for index in range(super_agent.n_agents)]
        score = 0
        step = 0
        evaluation = False
        coSharingCounter = 0
        # print("start")
        epsilon = 1.0 - (n_game / MAX_GAMES)
        if n_game > MAX_GAMES - 20:
            evaluation = True
        # loop through 20 times 600 simulation steps
        while not any(done):      
            # print("Epsilon :" + str(epsilon))
            actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)],epsilon,evaluation)
            
            actors_next_state, reward, done, info = env.step(actors_action)

            carFlowRate,bikeFlowRate,pedFlowRate,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,total_mean_speed_car,total_mean_speed_bike,total_mean_speed_ped,total_waiting_car_count,total_waiting_bike_count, total_waiting_ped_count,total_unique_car_count,total_unique_bike_count,total_unique_ped_count, \
                    car_occupancy,bike_occupancy,ped_occupancy,collision_count_bike,collision_count_ped,total_density_bike_lane,total_density_ped_lane, total_density_car_lane,Hinderance_bb,Hinderance_bp,Hinderance_pp,levelOfService = env.testAnalysisStats()
            
            rewardAgent_0, rewardAgent_1,rewardAgent_2 = env.rewardAnalysisStats()
            # rewardAgent_0, rewardAgent_1 = env.rewardAnalysisStats()
            writer.writerow([carFlowRate,bikeFlowRate,pedFlowRate,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,total_mean_speed_car,total_mean_speed_bike,total_mean_speed_ped,total_waiting_car_count,total_waiting_bike_count, total_waiting_ped_count,total_unique_car_count,total_unique_bike_count,\
                total_unique_ped_count,car_occupancy,bike_occupancy,ped_occupancy,collision_count_bike,collision_count_ped,total_density_bike_lane,total_density_ped_lane,total_density_car_lane,rewardAgent_0, rewardAgent_1,rewardAgent_2,Hinderance_bb,Hinderance_bp,Hinderance_pp,levelOfService])
            if step >= MAX_STEPS:
                done = True
            state = np.concatenate(actors_state)
            next_state = np.concatenate(actors_next_state)
            
            super_agent.replay_buffer.add_record(actors_state, actors_next_state, actors_action, state, next_state, reward, done)
            
            actors_state = actors_next_state
            
            if cosharing:
                coSharingCounter+=1
            # score += sum(reward)
            score += reward[0]
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
        # scores.append([np.mean(score), np.min(score), np.max(score)])
        # print_status(n_game, score, scores)
        # average of last 100 scores
        avg_score = np.mean(score_history[-100:])
        avg_score_list.append(avg_score)
        if n_game % PRINT_INTERVAL == 0 and n_game > 0:
            print('episode', n_game, 'average score {:.1f}'.format(avg_score))

        
        wandb.log({'Game number': super_agent.replay_buffer.n_games, '# Episodes': super_agent.replay_buffer.buffer_counter, 
                    "Average reward": round(np.mean(scores[-10:]), 2), \
                          "Time taken": round(time.time() - start_time, 2),\
                            "Cosharing Counter":coSharingCounter})
        
        if (n_game+1) % EVALUATION_FREQUENCY == 0 and super_agent.replay_buffer.check_buffer_size():
            actors_state = env.reset("Train")
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
            super_agent.save(env.pid)
            print("saved")
    


plot_scores([scores], ['ou'], save_as='results/normal.png')

plt.plot(avg_score_list)
plt.savefig('results/avgScore.jpg')
plt.show()
