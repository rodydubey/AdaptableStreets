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
from gym_sumo.envs.utils import generateFlowFiles

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

scenario = "Test 0"
# Test Case 0: Load pre-generated test flow files from a folder "testcase_0". This folder has 12*24 files.
# Each file simulates 5 mins or 300 simulation steps and it covers 24 hours period of randomly generated flow.


# Test Case 1: Load pre-generated test flow files from a folder "testcase_1". This folder has 12*24 files.
# Each file simulates 5 mins or 300 simulation steps and it covers 24 hours period using a real-world traffic flow distribution

number_files = 0
if scenario=="Test 0":
    lst = os.listdir("testcase_0/") # your directory path for test files
    number_files = len(lst)
    if number_files != 288:
        generateFlowFiles("Test 0")
    testResultFilePath = "result_test_0.csv"
else:
    lst = os.listdir("testcase_1/") # your directory path for test files
    number_files = len(lst)
    if number_files != 288:
        generateFlowFiles("Test 1")
    testResultFilePath = "result_test_0.csv"       

if PATH_LOAD_FOLDER is not None:
    print("loading weights")
    actors_state = env.reset(scenario)
    actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)],epsilon,evaluation)
    [super_agent.agents[index].target_actor(actors_state[index][None, :]) for index in range(super_agent.n_agents)]
    state = np.concatenate(actors_state)
    actors_action = np.concatenate(actors_action)
    [super_agent.agents[index].critic(state[None, :], actors_action[None, :]) for index in range(super_agent.n_agents)]
    [super_agent.agents[index].target_critic(state[None, :], actors_action[None, :]) for index in range(super_agent.n_agents)]
    super_agent.load()
   

    print(super_agent.replay_buffer.buffer_counter)
    print(super_agent.replay_buffer.n_games)
   

    with open(testResultFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Car_Flow_Rate','Bike_Flow_Rate','Ped_Flow_Rate','Car_Lane_Width','Bike_Lane_Width','Ped_Lane_Width','Co_Sharing','Total_mean_speed_car','Total_mean_speed_bike','Total_mean_speed_ped','Total_Waiting_car_count','Total_Waiting_bike_count','Total_Waiting_ped_count','Total_unique_car_count','Total_unique_bike_count','Total_unique_ped_count', \
             'Total_occupancy_car_Lane','Total_occupancy_bike_Lane','Total_occupancy_ped_Lane','Collision_count_bike','Collision_count_ped','total_density_bike_lane','total_density_ped_lane','total_density_car_lane','RewardAgent_0', 'RewardAgent_1','RewardAgent_2','Hinderance_bb','Hinderance_bp','Hinderance_pp','LevelOfService'])

        # for n_game in tqdm(range(1)):
            # if super_agent.replay_buffer.check_buffer_size():
       
       
        
        # while not any(done): 
        for step in tqdm(range(50)):
            done = [False for index in range(super_agent.n_agents)]
            actors_state = env.reset(scenario)
            step = 0
            score = 0
            while not any(done):  
                actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)],epsilon,evaluation)
                # if  step > 12:
                #     actors_action = [0.4,0.5,0]
                # else:
                #     actors_action = [0.4,0.5,1]
                actors_next_state, reward, done, info = env.step(actors_action)
                # super_agent.printCriticValues(actors_next_state,actors_action)
                state = np.concatenate(actors_state)
                next_state = np.concatenate(actors_next_state)
                actors_state = actors_next_state        
                score += reward[0]
                print(score)
                carFlowRate,bikeFlowRate,pedFlowRate,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,total_mean_speed_car,total_mean_speed_bike,total_mean_speed_ped,total_waiting_car_count,total_waiting_bike_count, total_waiting_ped_count,total_unique_car_count,total_unique_bike_count,total_unique_ped_count, \
                    car_occupancy,bike_occupancy,ped_occupancy,collision_count_bike,collision_count_ped,total_density_bike_lane,total_density_ped_lane, total_density_car_lane,Hinderance_bb,Hinderance_bp,Hinderance_pp,levelOfService = env.testAnalysisStats()
            
                rewardAgent_0, rewardAgent_1,rewardAgent_2 = env.rewardAnalysisStats()
                step += 1
                #reset the flow
                if step >= MAX_STEPS:
                    done = True                
                writer.writerow([carFlowRate,bikeFlowRate,pedFlowRate,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,total_mean_speed_car,total_mean_speed_bike,total_mean_speed_ped,total_waiting_car_count,total_waiting_bike_count, total_waiting_ped_count,total_unique_car_count,total_unique_bike_count,\
                total_unique_ped_count,car_occupancy,bike_occupancy,ped_occupancy,collision_count_bike,collision_count_ped,total_density_bike_lane,total_density_ped_lane,total_density_car_lane,rewardAgent_0, rewardAgent_1,rewardAgent_2,Hinderance_bb,Hinderance_bp,Hinderance_pp,levelOfService])
                if step >= MAX_STEPS:
                    break
        
        cumm_test_score_list.append(score)
        # #Plot Cummulative Queue Length for each vehicle type
        plt.plot(cumm_test_score_list)
        plt.show()

else:
    print("No model loaded")
