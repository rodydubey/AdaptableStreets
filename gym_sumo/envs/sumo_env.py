import gym
from gym import spaces
from gym.utils import seeding
from gym import spaces
import numpy as np
import math
from sumolib import checkBinary
import os, sys
sys.path.append('../') #allows loading of agent.py
from gym_sumo.envs.adapt_network import adaptNetwork, carLane_width_actions, bikeLane_width_actions
from gym_sumo.envs.adapt_route_file import adaptRouteFile
import xml.etree.ElementTree as ET
import math
from itertools import combinations, product
from utilss import get_space_dims

class Agent:
    def __init__(self, env, n_agent, edge_agent=None):
        """Dummy agent object"""
        self.edge_agent = edge_agent
        self.traci = env.traci
        self.env = env

        self.id = n_agent
        self.name = f'{edge_agent.edge_id} agent {self.id}'
    
    @property
    def edge_id(self):
        return self.edge_agent.edge_id
    
    def getState(self):
        """
        Retrieve the state of the network from sumo. 
        """
        agent_name = self.name
        # state = np.zeros(self._num_observation[agent_idx],dtype=np.float32)
        normalizeUniqueVehicleCount = 300
        laneWidthCar = self.traci.lane.getWidth(f'{self.edge_id}_2')
        laneWidthBike = self.traci.lane.getWidth(f'{self.edge_id}_1')
        laneWidthPed = self.traci.lane.getWidth(f'{self.edge_id}_0')
        nLaneWidthCar = np.interp(laneWidthCar, [0,12.6], [0,1])
        nLaneWidthBike = np.interp(laneWidthBike, [0,12.6], [0,1])
        nLaneWidthPed = np.interp(laneWidthPed, [0,12.6], [0,1])

        #E0 is for agent 0 and 1, #-E0 is for agent 2 and 3, #E1 is for agent 4 and 5, #-E1 is for agent 6 and 7
        #E2 is for agent 8 and 9, #-E2 is for agent 10 and 11, #E3 is for agent 12 and 13, #-E3 is for agent 14 and 15

        laneVehicleAllowedType = self.traci.lane.getAllowed(f'{self.edge_id}_0')
        if 'bicycle' in laneVehicleAllowedType:
            cosharing = 1
        else:
            cosharing = 0

        state = []

        if "agent 0" in agent_name: # car
            state_0 = laneWidthCar
            state_1 = laneWidthBike + laneWidthPed
            state_2 = self.edge_agent._total_occupancy_car_Lane			
            state_3 = self.edge_agent._total_density_car_lane
            
            state = [state_0, state_1, state_2, state_3]
            # print(state)
            # if state_2 > 1 or state_3 > 1:
            # 	print("Agent 0 observation out of bound")

        # xx = self.traci.lanearea.getLastIntervalOccupancy('det_0')	+ self.traci.lanearea.getLastIntervalOccupancy('det_1')
        # yy = self.traci.lanearea.getLastIntervalOccupancy('det_0_ped') + self.traci.lanearea.getLastIntervalOccupancy('det_1_ped')
        # ll = self.traci.lanearea.getLastIntervalMeanSpeed('det_0')	+ self.traci.lanearea.getLastIntervalMeanSpeed('det_1')
        # mm = self.traci.lanearea.getLastIntervalMeanSpeed('det_0_ped') + self.traci.lanearea.getLastIntervalMeanSpeed('det_1_ped')
        # aa = self.traci.lanearea.getLastIntervalVehicleNumber('det_0') + self.traci.lanearea.getLastIntervalVehicleNumber('det_1')
        # bb = self.traci.lanearea.getLastIntervalVehicleNumber('det_0_ped') + self.traci.lanearea.getLastIntervalVehicleNumber('det_1_ped')
    
        if "agent 1" in agent_name: # bike
            state_0 = laneWidthBike
            state_1 = laneWidthPed				
            state_2 = self.edge_agent._total_occupancy_bike_Lane		
            state_3 = self.edge_agent._total_occupancy_ped_Lane
            # if aa + bb:
            # 	state_2 = (aa)/(aa+bb)
            # 	state_3 = (xx*aa + yy*bb)/(aa+bb)/100
            # else:
            # 	state_2 = 0
            # 	state_3 = 0
            state = [state_0, state_1, state_2, state_3]
            # if state_2 > 1 or state_3 > 1:
            # 	print("Agent 1 observation out of bound")
        

        if "agent 2" in agent_name: 
            state_0 = laneWidthCar
            state_1 = laneWidthBike
            state_2 = laneWidthPed
            state_3 = self.edge_agent._total_occupancy_car_Lane	
            state_4 = self.edge_agent._total_density_car_lane
            state_5 = self.edge_agent._total_occupancy_bike_Lane
            state_6 = self.edge_agent._total_occupancy_ped_Lane
            state_7 = float(cosharing) #flag for cosharing on or off
            state_8 = float(np.abs(cosharing-1))
            state_9 = self.edge_agent._total_density_bike_lane
            state_10 = self.edge_agent._total_density_ped_lane
            # if aa + bb:
            # 	state_2 = (ll*aa + mm*bb)/(aa+bb)
            # 	state_3 = (xx*aa + yy*bb)/(aa+bb)/100
            # else:
            # 	state_2 = 0
            # 	state_3 = 0
            state = [state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8, state_9, state_10]
            # if state_1 > 1 or state_2 > 1:
            # 	print("Agent 2 observation out of bound")
        # print(state)
        return np.array(state)

    def getReward(self):
        # defaultCarLength = 5
        # defaultPedLength = 0.215
        # defaultBikeLength = 1.6
        agent_name = self.name
        laneVehicleAllowedType = self.traci.lane.getAllowed(f'{self.edge_id}_0')
        cosharing = False
        if 'bicycle' in laneVehicleAllowedType: 
            cosharing = True
        if "agent 0" in agent_name:
            carLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_2')
            if carLaneWidth < 3.2:
                reward = self.env._fatalPenalty
                self.done = True
            else:
                #occupancy reward. Lower Occupancy higher reward
                reward_occupancy_car = self.edge_agent._total_density_car_lane/10
                # reward_car_Stopped_count = self._total_count_waiting_car/(self.action_steps*10)
                # print("car stopped: " + str(reward_car_Stopped_count))
                reward = -(reward_occupancy_car)*1.5
                print("agent 0 reward: " + str(reward))
            

        elif "agent 1" in agent_name:
            bikeLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_1')
            pedLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_0')

            if cosharing == True:
                if (bikeLaneWidth + pedLaneWidth) < 2:
                    reward = self.env._fatalPenalty
                    self.done = True
                else:
                    reward = self.edge_agent._total_occupancy_ped_Lane*10/(self.env.action_steps) # as ped lane will count both waiting bikes and peds since the ped lane is coshared and bike lane width = 0
                    # print("bike + ped stopped in cosharing: " + str(reward))
                    reward = -reward*10
                    print("agent 1 reward: " + str(reward))
            else:
                if bikeLaneWidth < 1 or pedLaneWidth < 1:
                    reward = self.env._fatalPenalty
                    self.done = True
                else:
                    # reward = self.edge_agent._total_count_waiting_ped/(self.env.action_steps*10) + self.edge_agent._total_count_waiting_bike/(self.env.action_steps*10)
                    reward_occupancy_bike = self.edge_agent._total_occupancy_bike_Lane/self.env.action_steps
                    reward_occupancy_ped = self.edge_agent._total_occupancy_ped_Lane/self.env.action_steps
                    # print("bike + ped stopped: " + str(reward))
                    reward = -((reward_occupancy_bike+reward_occupancy_ped)/2)*100
                    print("agent 1 reward: " + str(reward))
                    reward = reward
        
        elif "agent 2" in agent_name:
            # collisionCount = self.edge_agent._collision_count_bike + self.edge_agent._collision_count_ped
            # if collisionCount > 50 and cosharing == True:
            # 	negative_reward_collision = -5
            # 	reward = negative_reward_collision
            # elif collisionCount < 50 and cosharing == True:
            # 	reward = +5
            # elif collisionCount < 50 and cosharing == False:
            # 	reward = -5
            # else:
            # 	negative_reward_collision = -0.01*collisionCount
            # 	reward = negative_reward_collision
            densityThreshold = 1
            
            if cosharing:
                if self.edge_agent._total_density_ped_lane > densityThreshold:
                    # reward = -(self.edge_agent._total_density_ped_lane - densityThreshold)/self.edge_agent._total_density_ped_lane
                    reward = -0.75
                elif self.edge_agent._total_density_ped_lane < densityThreshold:
                    # reward = (densityThreshold - self.edge_agent._total_density_ped_lane)/densityThreshold
                    reward = 0.75
            else:
                if (self.edge_agent._total_density_ped_lane + self.edge_agent._total_density_bike_lane) > 2*densityThreshold:
                    # reward = (self.edge_agent._total_density_ped_lane + self.edge_agent._total_density_bike_lane - 2*densityThreshold)/(self.edge_agent._total_density_ped_lane + self.edge_agent._total_density_bike_lane)
                    reward = 0.75
                elif (self.edge_agent._total_density_ped_lane + self.edge_agent._total_density_bike_lane) < 2*densityThreshold:
                    # reward = -(2*densityThreshold - self.edge_agent._total_density_ped_lane + self.edge_agent._total_density_bike_lane)/(2*densityThreshold)
                    reward = -0.75
            self.edge_agent.reward_agent_2 = reward

            # levelOfServiceThreshold_A = 5
            # levelOfServiceThreshold_B = 10
            # reward = 0.5
            # self.reward_agent_2 = reward
            # if cosharing:
            # 	if self._levelOfService > levelOfServiceThreshold_A:
            # 		reward = -reward
            # 	elif self._levelOfService < levelOfServiceThreshold_A:
            # 		reward = +reward
                
            # else:
            # 	if (self._levelOfService) > levelOfServiceThreshold_A:
            # 		reward = +reward
            # 	elif (self._levelOfService) < levelOfServiceThreshold_A:
            # 		# reward = self.env._fatalPenalty
            # 		# agent.done = True
            # 		reward = -reward

            # if cosharing == True:
            # 	# positive_reward_cosharing = +0.25
                
            # 	# negative_reward_collision = -0.01*collisionCount
                
                
            # 	# print("number of collision " + str(collisionCount))
            # 	# print("agent 2 reward: " + str(reward))
            # else:
            # 	# positive_reward_cosharing = -0.25  #25 collisions and below is
                
            # 	# print("agent 2 reward: " + str(reward))
            # # collisionCount = self._collision_count_bike/self.env.action_steps + self._collision_count_ped/self.action_steps
            # flowrate = float(self.FlowRateStatsFromRouteFile()[-1])/1200
            # if flowrate<0.5 and cosharing:
            # 	reward = 1
            # else:
            # 	reward = -1
            print("Agent 2 Reward :", self.edge_agent.reward_agent_2)

        return reward
    
class EdgeAgent:
    def __init__(self, env, edge_id):
        """Dummy agent object"""
        self.edge_id = edge_id
        self.traci = env.traci
        self.env = env
        self.resetAllVariables()
    
    def resetAllVariables(self):
        self._total_vehicle_passed_agent_0 = 0 
        self._total_pedestrian_passed_agent_1 = 0
        self._total_bike_passed_agent_1 = 0
        self._total_vehicle_on_lane_agent_0 = 0
        self._total_bike_on_lane_agent_1 = 0
        self._total_ped_on_lane_agent_1 = 0
        self._avg_ped_distance_agent_1 = 0
        self._avg_bike_distance_agent_1 = 0
        self._density = 0
        self._queue_Length_car_agent_0 = 0
        self._queue_Length_ped_agent_1 = 0
        self._queue_Length_bike_agent_1 = 0
        self._queue_Count_car_agent_0 = 0
        self._queue_Count_ped_agent_1 = 0
        self._queue_Count_bike_agent_1 = 0
        self._total_vehicle_passed_agent_2 = 0 
        # self._averageRewardStepCounter = 0
        self._unique_car_count_list = []
        self._unique_ped_count_list = []
        self._unique_bike_count_list = []
        self._total_unique_car_count = 0
        self._total_unique_bike_count = 0
        self._total_unique_ped_count = 0
        self._total_occupancy_car_Lane = 0
        self._total_waiting_time_car = 0
        self._total_waiting_time_bike = 0
        self._total_waiting_time_ped = 0
        self._total_count_waiting_car = 0
        self._total_waiting_time_bike = 0
        self._total_waiting_time_ped = 0
        self._total_mean_speed_car = 0
        self._total_mean_speed_bike = 0
        self._total_mean_speed_ped = 0
        self._total_count_waiting_bike = 0
        self._total_count_waiting_bike = 0
        self._total_count_waiting_ped = 0
        self._total_occupancy_bike_Lane = 0
        self._total_occupancy_ped_Lane = 0
        self._collision_count_bike = 0
        self._collision_count_ped = 0
        self._EmergencyBraking_count_bike = 0
        self._EmergencyBraking_count_ped = 0
        self._total_density_bike_lane = 0
        self._total_density_ped_lane = 0
        self._total_density_car_lane = 0
        self._total_hinderance_bike_bike = 0
        self._total_hinderance_bike_ped = 0
        self._total_hinderance_ped_ped = 0
        self._levelOfService = 0 
        
    def collectObservation(self):
        laneWidthCar = self.traci.lane.getWidth(f'{self.edge_id}_2')
        laneWidthBike = self.traci.lane.getWidth(f'{self.edge_id}_1')
        laneWidthPed = self.traci.lane.getWidth(f'{self.edge_id}_0')
        
        # The cosharing check is different due to the load and save state from previous episode. It may happen that the modified network i
        if laneWidthBike == 0:
            # print(str(coShare) + "--- YES Co-Sharing")
            disallowed3 = ['private', 'emergency', 'authority', 'passenger','army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
            disallowed3.append('bicycle')
            disallowed3.append('pedestrian')
            self.traci.lane.setDisallowed(f'{self.edge_id}_0',disallowed3)
            allowed = []
            allowed.append('bicycle')
            allowed.append('pedestrian')        
            self.traci.lane.setAllowed(f'{self.edge_id}_0',allowed)
            self.traci.lane.setDisallowed(f'{self.edge_id}_1', ["all"])			
        else: 
            # print(str(coShare) + "--- NO Co-Sharing")
            disallowed = ['private', 'emergency', 'passenger','authority', 'army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
            disallowed.append('pedestrian')
            self.traci.lane.setDisallowed(f'{self.edge_id}_1',disallowed)
            self.traci.lane.setAllowed(f'{self.edge_id}_1','bicycle')
            disallowed2 = ['private', 'emergency', 'passenger', 'authority', 'army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
            disallowed2.append('bicycle')
            self.traci.lane.setDisallowed(f'{self.edge_id}_0',disallowed2)
            self.traci.lane.setAllowed(f'{self.edge_id}_0','pedestrian')

        laneVehicleAllowedType = self.traci.lane.getAllowed(f'{self.edge_id}_0')
        if 'bicycle' in laneVehicleAllowedType:
            cosharing = True
        else:
            cosharing = False

        # record observatinos for each agent
        #Agent 0
        # # Count total number of unique cars on the car lane
        self._unique_car_count_list.extend(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_2'))
        # Count total occupancy of car lane in percentage
        self._total_occupancy_car_Lane += self.traci.lane.getLastStepOccupancy(f'{self.edge_id}_2')/laneWidthCar
        # Count total waiting time of the cars in the car lane
        self._total_waiting_time_car += self.traci.lane.getWaitingTime(f'{self.edge_id}_2')
        # Count total number of cars waiting in the car lane
        self._total_count_waiting_car += self.traci.lane.getLastStepHaltingNumber(f'{self.edge_id}_2')

        # test stats
        queue_length, queue_Count = self.env.getQueueLength(f'{self.edge_id}_2')
        self._queue_Length_car_agent_0 += queue_length

        queue_length, queue_Count = self.env.getQueueLength(f'{self.edge_id}_1')
        self._queue_Length_bike_agent_1 += queue_length

        queue_length, queue_Count = self.env.getQueueLength(f'{self.edge_id}_0')
        self._queue_Length_ped_agent_1 += queue_length



        self._total_waiting_time_bike += self.traci.lane.getWaitingTime(f'{self.edge_id}_1') 
        self._total_waiting_time_ped += self.traci.lane.getWaitingTime(f'{self.edge_id}_0') 
        # test stats

        #Returns the mean speed of vehicles that were on this lane within the last simulation step [m/s]
        self._total_mean_speed_car += self.traci.lane.getLastStepMeanSpeed(f'{self.edge_id}_2')
        self._total_mean_speed_bike += self.traci.lane.getLastStepMeanSpeed(f'{self.edge_id}_1')
        self._total_mean_speed_ped += self.traci.lane.getLastStepMeanSpeed(f'{self.edge_id}_0')

        # Count total number of bikes waiting in the bike lane
        self._total_count_waiting_bike += self.traci.lane.getLastStepHaltingNumber(f'{self.edge_id}_1')
        # Count total number of peds waiting in the ped lane
        self._total_count_waiting_ped += self.traci.lane.getLastStepHaltingNumber(f'{self.edge_id}_0')


        # carCollisionCount, bikeCollisionCount, pedCollisionCount = self.env.getAllCollisionCount()
        # carBrakeCount, bikeBrakeCount, pedBrakeCount = self.env.getAllEmergencyBrakingCount()
        self._total_density_bike_lane += self.env.getDensityOfALaneID(f'{self.edge_id}_1')
        self._total_density_ped_lane += self.env.getDensityOfALaneID(f'{self.edge_id}_0')
        self._total_density_car_lane += self.env.getDensityOfALaneID(f'{self.edge_id}_2')
        # self._EmergencyBraking_count_bike += bikeBrakeCount
        # self._EmergencyBraking_count_bike += bikeBrakeCount
        # self._EmergencyBraking_count_ped += pedBrakeCount
        
        if cosharing:
            #Agent 1
            # Count total number of unique pedestrian on the ped lane
            # self._total_unique_bike_count += 0 # because this lane width is merged into pedestrian
            # Count total number of unique pedestrian + bike on the ped lane
            # self._total_unique_ped_count += self.getUniquePedCount()
            self._unique_ped_count_list.extend(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_0'))
            self._unique_bike_count_list.extend(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_1'))
            self._total_occupancy_bike_Lane += 0
            # Count total occupancy of ped lane in percentage
            self._total_occupancy_ped_Lane += self.traci.lane.getLastStepOccupancy(f'{self.edge_id}_0')/laneWidthPed

            #Agent 2
            # self._collision_count_bike += bikeCollisionCount
            # self._collision_count_ped += pedCollisionCount

            if self.env._sumo_step % 10 == 0 and ("Test" in self.env._scenario):
                h_b_b, h_b_p, h_p_p =  self.getHinderenaceWhenCosharing(f'{self.edge_id}_0')
                self._total_hinderance_bike_bike += h_b_b
                self._total_hinderance_bike_ped += h_b_p
                self._total_hinderance_ped_ped += h_p_p
            # print("hinderance bike with ped : " + str(hinderance))

        else:
            #Agent 1
            self._unique_bike_count_list.extend(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_1'))
            self._unique_ped_count_list.extend(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_0'))
            # Count total occupancy of bike lane in percentage
            self._total_occupancy_bike_Lane += self.traci.lane.getLastStepOccupancy(f'{self.edge_id}_1')/laneWidthBike
            # Count total occupancy of ped lane in percentage
            self._total_occupancy_ped_Lane += self.traci.lane.getLastStepOccupancy(f'{self.edge_id}_0')/laneWidthPed
            if self.env._sumo_step % 10 == 0 and ("Test" in self.env._scenario):
                self._total_hinderance_bike_bike += self.getHinderenace(f'{self.edge_id}_1',"bike_bike")
                self._total_hinderance_ped_ped += self.getHinderenace(f'{self.edge_id}_0',"ped_ped")

            #Agent 2
            # self._collision_count_bike += bikeCollisionCount
            # self._collision_count_ped += pedCollisionCount


    def LevelOfService(self,coSharing):
        # It is a function of lane width, total vehicle number, hindrance_bb,hinderence_cc,hindrance_bc}
        self.w_lane_width = 0.3
        self.w_total_occupancy = 0.3
        _total_hinderance_ped_ped = self._total_hinderance_ped_ped/30
        _total_hinderance_bike_bike = self._total_hinderance_bike_bike/30
        _total_hinderance_bike_ped = self._total_hinderance_bike_ped/30
        _avg_waiting_time_car = self._total_waiting_time_car/(100*300)

        if coSharing:
            self.w_hinderance_b_b = 0.1
            self.w_hinderance_b_p = 0.2
            self.w_hinderance_p_p = 0.1
            laneID = f'{self.edge_id}_0'
            laneWidth = self.traci.lane.getWidth(laneID)/12.6
            los = -self.w_lane_width*laneWidth + self.w_total_occupancy*self._total_occupancy_ped_Lane  + self.w_hinderance_b_b*_total_hinderance_bike_bike + \
                 self.w_hinderance_b_p*_total_hinderance_bike_ped + self.w_hinderance_p_p*_total_hinderance_ped_ped

        else:
            self.w_hinderance_b_b = 0.2
            self.w_hinderance_b_p = 0
            self.w_hinderance_p_p = 0.2
            pedLaneID = f'{self.edge_id}_0'
            bikeLaneID = f'{self.edge_id}_1'
            pedLaneWidth = self.traci.lane.getWidth(pedLaneID)/12.6
            bikeLaneWidth = self.traci.lane.getWidth(bikeLaneID)/12.6
            los_ped_Lane = -self.w_lane_width*pedLaneWidth + self.w_total_occupancy*self._total_occupancy_ped_Lane  + self.w_hinderance_p_p*_total_hinderance_ped_ped
            los_bike_Lane = -self.w_lane_width*bikeLaneWidth + self.w_total_occupancy*self._total_occupancy_bike_Lane  + self.w_hinderance_b_b*_total_hinderance_bike_bike
            los = (los_ped_Lane + los_bike_Lane)/2

        total_los = _avg_waiting_time_car + los

        if total_los <0:
            total_los = 0
        return total_los

    def getTestStats(self):
        #returns average waiting time for car, cycle, ped
        #returns average queue length for car, cycle, ped
        #returns LOS per step

        avg_waiting_time_car = self._total_waiting_time_car/self.env.action_steps
        avg_queue_length_car = self._queue_Length_car_agent_0/self.env.action_steps

        avg_waiting_time_bike = self._total_waiting_time_bike/self.env.action_steps
        avg_queue_length_bike = self._queue_Length_bike_agent_1/self.env.action_steps

        avg_waiting_time_ped = self._total_waiting_time_ped/self.env.action_steps
        avg_queue_length_ped = self._queue_Length_ped_agent_1/self.env.action_steps
        los = self._levelOfService
        laneVehicleAllowedType = self.traci.lane.getAllowed(f'{self.edge_id}_0')
        if 'bicycle' in laneVehicleAllowedType:
            cosharing = True
        else:
            cosharing = False

        headers = ['avg_waiting_time_car', 'avg_waiting_time_bike', 'avg_waiting_time_ped',
                   'avg_queue_length_car', 'avg_queue_length_bike', 'avg_queue_length_ped',
                   'los', "Reward_Agent_2", "cosharing", 'edge_id']
        values = [avg_waiting_time_car, avg_waiting_time_bike, avg_waiting_time_ped,
                  avg_queue_length_car, avg_queue_length_bike, avg_queue_length_ped,
                  los, self.reward_agent_2, cosharing, self.edge_id]
        return headers, values


    def testAnalysisStats(self):
        bikeLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_1')
        pedlLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_0')
        carLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_2')
        laneVehicleAllowedType = self.traci.lane.getAllowed(f'{self.edge_id}_0')
        if 'bicycle' in laneVehicleAllowedType:
            cosharing = 1
        else:
            cosharing = 0

        
        self._carFlow,self._bikeFlow,self._pedFlow = self.FlowRateStatsFromRouteFile()
        
        output_headers = ['edge_id', 'Car_Flow_Rate', 'Bike_Flow_Rate', 'Ped_Flow_Rate', 'Car_Lane_Width', 'Bike_Lane_Width', 
                          'Ped_Lane_Width', 'Co_Sharing', 'Total_occupancy_car_Lane', 'Total_occupancy_bike_Lane', 
                          'Total_occupancy_ped_Lane', 'total_density_bike_lane', 'total_density_ped_lane', 
                          'total_density_car_lane', 'LevelOfService']
        output_vals = [self.edge_id, self._carFlow, self._bikeFlow, self._pedFlow, carLaneWidth, bikeLaneWidth, pedlLaneWidth, cosharing, 
                    #    self._total_mean_speed_car, self._total_mean_speed_bike, self._total_mean_speed_ped, 
                    #    self._total_count_waiting_car, self._total_count_waiting_bike, self._total_count_waiting_ped, self._total_unique_car_count, self._total_unique_bike_count, self._total_unique_ped_count,
                       self._total_occupancy_car_Lane, self._total_occupancy_bike_Lane, self._total_occupancy_ped_Lane, 
                    #    self._collision_count_bike, self._collision_count_ped, 
                       self._total_density_bike_lane, self._total_density_ped_lane, self._total_density_car_lane,
                    #    self._total_hinderance_bike_bike, self._total_hinderance_bike_ped, self._total_hinderance_ped_ped, 
                       self._levelOfService]
        # carFlowRate,bikeFlowRate,pedFlowRate,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,
        # total_mean_speed_car,total_mean_speed_bike,total_mean_speed_ped,total_waiting_car_count,total_waiting_bike_count, total_waiting_ped_count,total_unique_car_count,total_unique_bike_count,total_unique_ped_count, \
        
        # car_occupancy,bike_occupancy,ped_occupancy,collision_count_bike,collision_count_ped,total_density_bike_lane,total_density_ped_lane, total_density_car_lane,Hinderance_bb,Hinderance_bp,Hinderance_pp,levelOfService
        
        # carFlowRate,bikeFlowRate,pedFlowRate,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,\
        # car_occupancy,bike_occupancy,ped_occupancy,total_density_bike_lane,total_density_ped_lane,total_density_car_lane,rewardAgent_0, rewardAgent_1,rewardAgent_2,levelOfService
        return output_headers, output_vals
    
    def FlowRateStatsFromRouteFile(self):
        tree = ET.parse(self.env._routeFileName)
        root = tree.getroot()
        vehsPerHour = 0
        bikesPerHour = 0
        pedsPerHour = 0
        for flows in root.iter('flow'):		
            if flows.attrib['id'] == f"{self.edge_id}_f_2":
                vehsPerHour = flows.attrib['vehsPerHour']
            elif flows.attrib['id'] == f"{self.edge_id}_f_1":
                bikesPerHour = flows.attrib['vehsPerHour']
            elif flows.attrib['id'] == f"{self.edge_id}_f_0":
                pedsPerHour = flows.attrib['vehsPerHour']

        return vehsPerHour,bikesPerHour,pedsPerHour

    def getHinderenaceWhenCosharing(self,laneID):
        h_b_b = 0
        h_b_p = 0
        h_p_p = 0
        bikeList = []
        pedList = []
        
        allVehicles = self.traci.lane.getLastStepVehicleIDs(laneID)			
        if len(allVehicles) > 1:
            for veh in allVehicles:
                x = veh.rsplit("_",1)
                vehID = x[1].split(".",1)
                if vehID[0]=="1":
                    bikeList.append(veh)
                elif vehID[0]=="0":
                    pedList.append(veh)

        pos_bikes = [(self.traci.vehicle.getPosition(bike)) for bike in bikeList]
        pos_peds = [(self.traci.vehicle.getPosition(ped)) for ped in pedList]

        for bike, pos_bike in enumerate(pos_bikes):
            pos_bike_x,pos_bike_y = pos_bike
            for bb, pos_bb in enumerate(pos_bikes):
                if bike != bb:
                    pos_bb_x,pos_bb_y = pos_bb

                    distance = self.traci.simulation.getDistance2D(pos_bike_x,pos_bike_y,pos_bb_x,pos_bb_y)
                    if distance < 1:
                        h_b_b +=1
        for bike, pos_bike in enumerate(pos_bikes):
            pos_bike_x,pos_bike_y = pos_bike
            for ped, pos_ped in enumerate(pos_peds):
                if bike != ped:
                    pos_ped_x,pos_ped_y = pos_ped

                    distance = self.traci.simulation.getDistance2D(pos_bike_x,pos_bike_y,pos_ped_x,pos_ped_y)
                    if distance < 1:
                        h_b_p +=1
        for ped, pos_ped in enumerate(pos_peds):
            pos_ped_x,pos_ped_y = pos_ped
            for pp, pos_pp in enumerate(pos_peds):
                if ped != pp:
                    pos_pp_x,pos_pp_y = pos_pp

                    distance = self.traci.simulation.getDistance2D(pos_ped_x,pos_ped_y,pos_pp_x,pos_pp_y)
                    if distance < 1:
                        h_p_p +=1
        return h_b_b,h_b_p,h_p_p

    def getHinderenace(self,laneID,betweenVehicleType):
        hinderance = 0
        bikeList = []
        pedList = []
        allVehicles = self.traci.lane.getLastStepVehicleIDs(laneID)
        if len(allVehicles) > 1:
            for veh in allVehicles:
                x = veh.rsplit("_",1)
                vehID = x[1].split(".",1)
                if vehID[0]=="1":
                    bikeList.append(veh)
                elif vehID[0]=="0":
                    pedList.append(veh)
        pos_bikes = [(self.traci.vehicle.getPosition(bike)) for bike in bikeList]
        pos_peds = [(self.traci.vehicle.getPosition(ped)) for ped in pedList]

        if betweenVehicleType == "bike_bike":
            for bike, posxy in enumerate(pos_bikes):
                pos_bike_x,pos_bike_y = posxy
                for bb, posxy_bb in enumerate(pos_bikes):
                    if bike != bb:
                        pos_bb_x,pos_bb_y = posxy_bb
                        distance = self.traci.simulation.getDistance2D(pos_bike_x,pos_bike_y,pos_bb_x,pos_bb_y)
                        if distance < 1:
                            hinderance +=1

        elif betweenVehicleType == "bike_ped":
            for bike, posxy in enumerate(pos_bikes):
                pos_bike_x,pos_bike_y = posxy
                for ped, posxy_ped in enumerate(pos_peds):
                    if bike != ped:
                        pos_ped_x,pos_ped_y = posxy_ped
                        distance = self.traci.simulation.getDistance2D(pos_bike_x,pos_bike_y,pos_ped_x,pos_ped_y)
                        if distance < 1:
                            hinderance +=1		

        elif betweenVehicleType == "ped_ped":
            for ped, posxy in enumerate(pos_peds):
                pos_ped_x,pos_ped_y = posxy
                for pp, posxy_pp in enumerate(pos_peds):
                    if ped != pp:
                        pos_pp_x,pos_pp_y = posxy_pp
                        distance = self.traci.simulation.getDistance2D(pos_ped_x,pos_ped_y,pos_pp_x,pos_pp_y)
                        if distance < 1:
                            hinderance +=1	

            hinderance = 0
        
        return hinderance
    
class SUMOEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
    # small_lane_ids = ['E0_2','E0_1','E0_0']

    def __init__(self,reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True,mode='gui',
                 edges=['E0', '-E1','-E2', '-E3'], simulation_end=36000,
                 joint_agents=False):
        self.pid = os.getpid()
        self.sumoCMD = []
        self.modeltype = 'model'
        self.joint_agents = joint_agents
        self.generatedFiles = []
        self._simulation_end = simulation_end
        self._mode = mode
        # self._seed(40)
        np.random.seed(42)
        self.sumo_seed = np.random.randint(69142)
        self.counter = 2
        self.withGUI = mode=='gui'
        self.traci = self.initSimulator(self.withGUI, self.pid)
        self._sumo_step = 0
        self._episode = 0
        self.agent_types = []
        self._flag = True       
        self._weightCar = 1
        self._weightPed = 10
        self._weightBike = 3
        self._weightBikePed = 2 
        # self._gamma = 0.75
        self._slotId = 1
        self.timeOfHour = 1
        self.base_netfile = "environment/intersection.net.xml"
        self._routeFileName = "environment/intersection_Slot_1.rou.xml" # default name 
        self._max_steps = 24000
        self._slot_duration = 1200
        self._max_slots = 3
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self.action_steps = 300
        self.sumo_running = False
        self.viewer = None
        self.firstTimeFlag = True
        # self.observation = self.reset()
        self._slotId = 0
        self._carQueueLength = 0
        self._bikeQueueLength = 0
        self._pedQueueLength = 0
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.shared_reward = True
        self._fatalErroFlag = False
        self.time = 0
        self._fatalPenalty = -20
        self.lanes_length = 100
        self._carFlow = 0
        self._pedFlow = 0
        self._bikeFlow = 0

        # self._scenario = "Train"
        # set required vectorized gym env property
        self.edges = edges
        self._num_lane_agents = 3
        
        # configure spaces
        self.edge_agents = [EdgeAgent(self, edge_id) for edge_id in self.edges]
        if self.joint_agents:
            num_agent_factor = len(self.edge_agents)
        else:
            num_agent_factor = 1
        self.n = self._num_lane_agents*num_agent_factor
        self._num_observation = [len(Agent(self, i, self.edge_agents[0]).getState()) for i in range(self._num_lane_agents)]*num_agent_factor
        self._num_actions = [len(carLane_width_actions), len(bikeLane_width_actions),2]*num_agent_factor
        self.action_space = []
        self.observation_space = []
       
        for i in range(self.n):
            if self._num_actions[i]==1:
                self.action_space.append(spaces.Box(low=0, high=+1, shape=(1,))) # alpha value
            else:
                self.action_space.append(spaces.Discrete(self._num_actions[i]))
            # observation space
            self.observation_space.append(spaces.Box(low=0, high=+1, shape=(self._num_observation[i],)))
            self.agent_types.append("cooperative")
            # if agent.name == "agent 0":
            # 	self.observation_space.append(spaces.Box(low=0, high=+1, shape=(self._num_observation,)))
            # else:
            # 	self.observation_space.append(spaces.Box(low=0, high=+1, shape=(self._num_observation+1,)))
        self.action_space = spaces.Tuple(self.action_space)
        self.observation_space = spaces.Dict(spaces={f'E0 agent {i}': o_space 
                                                     for i, o_space in enumerate(self.observation_space)})
        self.agents = self.createNAgents(self.edge_agents)
        # self.action_space = spaces.Box(low=np.array([0]), high= np.array([+1])) # Beta value 
        # self.observation_space = spaces.Box(low=0, high=1, shape=(np.shape(self.observation)))
        # self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32))

    def set_run_mode(self, mode): 
        if mode in ['none', 'Test']:
            self._scenario = "Test"
        elif mode == 'Test Single Flow':
            self._scenario = "Test Single Flow"
        else:
            self._scenario = "Train"

    def createNAgents(self, edge_agents):

        agents = []
        for j, edge_agent in enumerate(edge_agents):
            edge_id = edge_agent.edge_id
            # edge_agents.append(edge_agent)
            for agent_id in range(0,self._num_lane_agents): #fix this number 3
                agents.append(Agent(self, agent_id, edge_agent))
        return agents
    
    # Edge E0 - Agent 1 and Agent 2
    # Edge E0 - Agent 1 and Agent 2
    # Edge E0 - Agent 1 and Agent 2

    
    # def getState(self, agent):
    #     state = agent.getState()
    #     return state
    
    def make_action(self,actions):
        agent_actions = []
        for j, edge_id in enumerate(self.edges):
            for i in range(0,self._num_lane_agents): #fix this number 3
                index = np.argmax(actions[j*self._num_lane_agents+i])
                agent_actions.append(index)
        return agent_actions

    def _collect_waiting_times_cars(self,laneID):
        """
        Retrieve the waiting time of every car in the incoming roads
        """

        # car_list = self.traci.vehicle.getIDList()
        nCars= self.traci.lane.getLastStepVehicleIDs(laneID)
        waiting_times = {}
        avg_waiting_time = 0
        if len(nCars) > 0:
            for car_id in nCars:
                wait_time = self.traci.vehicle.getAccumulatedWaitingTime(car_id)
                waiting_times[car_id] = wait_time
                
            avg_waiting_time = sum(waiting_times.values())/len(nCars)
            temp_total_wait_time = self.traci.lane.getWaitingTime(laneID)
            # print(total_waiting_time)
            # print(temp_total_wait_time)
        
        return avg_waiting_time

    def readRouteFile(self,name):
        tree = ET.parse(self._routeFileName)
        root = tree.getroot()
        vehsPerHour = 0
        if name == "car":
            vehType = "f_2"
        elif name == "ped":
            vehType = "f_0"
        elif name == "bike":
            vehType = "f_1"
        for flows in root.iter('flow'):		
            if flows.attrib['id'] == vehType:
                vehsPerHour = flows.attrib['vehsPerHour']

        return vehsPerHour



    # def getActionStateAfterWarmUpPeriod(self):
    # 	#record observatinos for each agent
    # 	obs_n = []
    # 	for agent in self.agents:
    # 		obs_n.append(self._get_obs(agent))
    # 	return obs_n
    
    # def outPutLaneWidth(self):
    #     for ilane in range(0, 3):
    #         lane_id = self.small_lane_ids[ilane]
    #         tempLaneWidth = self.traci.lane.getWidth(lane_id)
    #         print(lane_id + ":" + str(tempLaneWidth))

    # def get_lane_queue(self,lane,min_gap):
    #     lanes_queue = self.traci.lane.getLastStepHaltingNumber(lane) / (self.lanes_length/ (min_gap + self.traci.lane.getLastStepLength(lane)))

    def reset(self, *args):		
        self._sumo_step = 0
        # self._scenario = "Train"
        for edge_agent in self.edge_agents:
            edge_agent.resetAllVariables()
        if self._scenario=="Train":
            self._slotId = np.random.randint(1,120)
            #Adapt Route File for continous change
            # self._slotId = 3 # temporary
            # adaptRouteFile(self._slotId, self.pid)
            # if self._slotId < 27:
            # 	self._slotId += 1 
            # else:
            # 	self._slotId = 1
            self._routeFileName = "environment/intersection_Slot_" + str(self._slotId) + ".rou.xml"
            print(self._routeFileName)
        elif self._scenario=="Test":
            self._slotId = self.timeOfHour
            # self._slotId = 35
            if len(self.edges)==1:
                print("Testing")
                self._routeFileName = "testcase_0/two/intersection_Slot_" + str(self._slotId) + ".rou.xml"
            elif len(self.edges)==4:
                print("Testing 4wayflow")
                self._routeFileName = "testcase_0/4way/intersection_Slot_" + str(self._slotId) + ".rou.xml"
            print(self._routeFileName)
            self.timeOfHour +=1
        elif self._scenario=="Test Single Flow":
            self.timeOfHour = 1
            if len(self.edges)==1:
                print("Testing")
                self._routeFileName = "testcase_0/daytest/flows.rou.xml"
            elif len(self.edges)==4:
                print("Testing 4wayflow")
                self._routeFileName = "testcase_0/4way/flows.rou.xml"
            print(self._routeFileName)
        else:
            self._slotId = np.random.randint(1, 288)
            self._routeFileName = "testcase_1/intersection_Slot_" + str(self._slotId) + ".rou.xml"
            print(self._routeFileName)
        
        obs_n = {}
        # self.traci.load(['-n', 'environment/intersection.net.xml', '-r', self._routeFileName, "--start"]) # should we keep the previous vehicle
        if self.firstTimeFlag:
            self.traci.load(self.sumoCMD + ['-n', 'environment/intersection.net.xml', '-r', self._routeFileName])
            # if self._scenario=="Train":
            while self._sumo_step <= self.action_steps:
                self.traci.simulationStep() 		# Take a simulation step to initialize
                self.collectObservation()
                self._sumo_step +=1
            #     self.firstTimeFlag = False
        else:
            modified_netfile = f'environment/intersection2_{self.pid}.net.xml'
            self.traci.load(self.sumoCMD + ['-n', modified_netfile, '-r', self._routeFileName])
        
        
        
                
        #record observatinos for each agent
        for edge_agent in self.edge_agents:
            edge_agent._total_unique_car_count = len(np.unique(np.array(edge_agent._unique_car_count_list)))
            edge_agent._total_unique_bike_count = len(np.unique(np.array(edge_agent._unique_bike_count_list)))
            edge_agent._total_unique_ped_count = len(np.unique(np.array(edge_agent._unique_ped_count_list)))
        for agent in self.agents:   
            agent.done = False
            # obs_n.append(self._get_obs(agent))
            obs_n[agent.name] = self._get_obs(agent)
        return obs_n

    # get observation for a particular agent
    def _get_obs(self, agent):
        state = agent.getState()
        return state
        # return self.getState(agent)

    # def _observation(self,agent):
    # 	return self.getState(agent)


    def getQueueLength(self,laneID):
        allVehicles = self.traci.lane.getLastStepVehicleIDs(laneID)
        queueCount = 0
        if len(allVehicles) > 1:
            lastMaxPos = 60
            for veh in allVehicles:
                speed = self.traci.vehicle.getSpeed(veh)
                pos_p_x = self.traci.vehicle.getPosition(veh)[0]
                if speed < 0.1 and pos_p_x > -35:					
                    # pos_p_y = self.traci.vehicle.getPosition(veh)[1]
                    queueCount +=1
                    if pos_p_x < lastMaxPos:
                        lastMaxPos = pos_p_x

            queueLength = 60 - lastMaxPos
        else:
            queueLength = 0
        return queueLength, queueCount

    def dist(self,p1,p2):
        (x1, y1), (x2, y2) = p1, p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # function to compute spacing between all pedestrians 
    def interDistanceBetweenPed(self,PedLaneID):
        #loop through all pedestrian on a lane
        allPeds = self.traci.lane.getLastStepVehicleIDs(PedLaneID)
        if len(allPeds) > 1:
            pos_p_x = []
            pos_p_y = []
            for ped in allPeds:
                pos_p_x.append(self.traci.vehicle.getPosition(ped)[0])
                pos_p_y.append(self.traci.vehicle.getPosition(ped)[1])
            
            points = list(zip(pos_p_x,pos_p_y))
            distances = [math.dist(p1, p2) for p1, p2 in combinations(points, 2)]
            avg_distance_pedestrian = sum(distances) / len(distances)
            return avg_distance_pedestrian
        else:
            return 0

    def getDensityOfALaneID(self,laneID):
        num = self.traci.lane.getLastStepVehicleNumber(laneID)
        length = self.traci.lane.getLength(laneID)
        width = self.traci.lane.getWidth(laneID)
        if width == 0:
            return 0
        density = num / (length * width)
        normDensity = density
        return normDensity
        
    # function to compute spacing between all bikes 
    def interDistanceBetweenBikes(self,BikeLaneID):
        #loop through all pedestrian on a lane
        allBikes = self.traci.lane.getLastStepVehicleIDs(BikeLaneID)
        if len(allBikes) > 1:
            pos_p_x = []
            pos_p_y = []
            for bike in allBikes:
                pos_p_x.append(self.traci.vehicle.getPosition(bike)[0])
                pos_p_y.append(self.traci.vehicle.getPosition(bike)[1])
            
            points = list(zip(pos_p_x,pos_p_y))
            distances = [math.dist(p1, p2) for p1, p2 in combinations(points, 2)]
            avg_distance_bikes = sum(distances) / len(distances)
            return avg_distance_bikes
        else:
            return 0

    # get reward for a particular agent
    def _get_reward(self,agent):
        # for agent in self.agents:
        reward = agent.getReward()
        return reward

            
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_done(self, agent):  
        return agent.done

    # get info used for benchmarking
    def _get_info(self, agent):
        return {}

    #Count number of unique cars on the car lane
    def getUniqueCarCount(self):
        carsCountList = self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_2')
        self._unique_car_count_list.extend(carsCountList)
        # self._unique_car_count_list.append(carsCountList)
        num_values = len(np.unique(np.array(self._unique_car_count_list)))
        return num_values

    def getUniquePedCount(self):
        pedCountList = self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_0')
        self._unique_ped_count_list.extend(pedCountList)
        num_values = len(np.unique(np.array(self._unique_ped_count_list)))
        return num_values

    def getUniqueBikeCount(self):
        bikeCountList = self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_1')
        self._unique_bike_count_list.extend(bikeCountList)
        num_values = len(np.unique(np.array(self._unique_bike_count_list)))
        return num_values

    # def getAllEmergencyBrakingCount(self):		
    #     allBrakingVehicleIDList = self.traci.simulation.getEmergencyStoppingVehiclesIDList()
    #     bikeBrakeCounter = 0
    #     pedBrakeCounter = 0
    #     carBrakeCounter = 0
    #     for veh in allBrakingVehicleIDList:
    #         # print(veh)
    #         x = veh.split("_",2)
    #         vehID = x[1].split(".",1)
    #         if vehID[0]=="1":
    #             bikeBrakeCounter +=1
    #         elif vehID[0]=="0":
    #             pedBrakeCounter +=1
    #         elif vehID[0]=="2":
    #             carCollisionCounter +=1

    #     return carBrakeCounter,bikeBrakeCounter,pedBrakeCounter
    # def getAllCollisionCount(self):		
    #     allCollidingVehicleIDList = self.traci.simulation.getCollidingVehiclesIDList()
    #     bikeCollisionCounter = 0
    #     pedCollisionCounter = 0
    #     carCollisionCounter = 0
    #     for veh in allCollidingVehicleIDList:
    #         # print(veh)
    #         x = veh.split("_",2)
    #         vehID = x[1].split(".",1)
    #         if vehID[0]=="1":
    #             bikeCollisionCounter +=1
    #         elif vehID[0]=="0":
    #             pedCollisionCounter +=1
    #         elif vehID[0]=="2":
    #             carCollisionCounter +=1

    #     return carCollisionCounter,bikeCollisionCounter,pedCollisionCounter
    
    def collectObservation(self):
        for edge_agent in self.edge_agents:
            edge_agent.collectObservation()


    def step(self,action_n):
        obs_n = {}
        reward_n = []
        done_n = []
        info_n = {'n':[]}
        rewardFlag = False
        actionFlag = True
        for agent in self.agents:
            agent.done = False
        
        self._sumo_step = 0
        # adaptRouteFile(self._slotId, self.pid)
        # self.traci.load(self.sumoCMD + ['-n', 'environment/intersection.net.xml', '-r', self._routeFileName])
        #simulating a warm period of N=self.action_steps  and then recording the state, action, reward tuple. 
        # bikeLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_1')
        # pedlLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_0')
        # carLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_2')
        
        
        
        #set action for each agents
        if actionFlag == True:
            temp_action_dict = {}
            action_space_dict = {}
            simple_actions = self.make_action(action_n)
            # print(simple_actions, [len(i) for i in action_n])
            # print([(agent.name, agent.edge_id) for agent in self.agents])
            # for i, edge_agent in enumerate(self.edge_agents):
            #     for j in range(self.n):
            for i, agent in enumerate(self.agents):
                # index = np.argmax(action_n[i])
                # temp_action_dict[agent.name] = int(index)
                #for continous action space
                # temp_action_dict[(agent.name, edge_agent.edge_id)] = simple_actions[i*self.n+j]
                temp_action_dict[(agent.name, agent.edge_id)] = simple_actions[i]
                # action_space_dict[(agent.name, agent.edge_id)] = self.action_space[i//self.n + i%(self.n)]
                # action_space_dict[(agent.name, agent.edge_id)] = self.action_space[i]
                # if agent.name == "agent 2":
                #     self.coShareValue = simple_actions[i]
            self._set_action(temp_action_dict, self.modeltype)
            actionFlag = False
            # if 'Test' in self._scenario:
            self.warmup()
        if self._scenario in ["Train", "Test", "Test Single Flow"]:
            #reset all variables
            for edge_agent in self.edge_agents:
                edge_agent.resetAllVariables()
                laneVehicleAllowedType = self.traci.lane.getAllowed(f'{edge_agent.edge_id}_0')
                if 'bicycle' in laneVehicleAllowedType:
                    edge_agent.cosharing = True
                else:
                    edge_agent.cosharing = False
            while self._sumo_step <= self.action_steps:
                # advance world state
                self.traci.simulationStep()
                self._sumo_step +=1
                self.collectObservation()
            
            for edge_agent in self.edge_agents:
                edge_agent._total_unique_car_count = len(np.unique(np.array(edge_agent._unique_car_count_list).flatten()))
                edge_agent._total_unique_bike_count = len(np.unique(np.array(edge_agent._unique_bike_count_list).flatten()))
                edge_agent._total_unique_ped_count = len(np.unique(np.array(edge_agent._unique_ped_count_list).flatten()))

                edge_agent._levelOfService = edge_agent.LevelOfService(edge_agent.cosharing)
                print("Level of Service : " + str(edge_agent.LevelOfService(edge_agent.cosharing)))
                print("Cosharing :", edge_agent.cosharing)
                
                print("Density :", (edge_agent._total_density_ped_lane + edge_agent._total_density_bike_lane))
            
        # 	if 'bicycle' in laneVehicleAllowedType:
        # 		cosharing = True
        # 	else:
        # 		cosharing = False
        # 	# record observatinos for each agent
        # 	#Agent 0
        # 	# # Count total number of unique cars on the car lane
        # 	self._unique_car_count_list.append(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_2'))
        # 	# Count total occupancy of car lane in percentage
        # 	self._total_occupancy_car_Lane += self.traci.lane.getLastStepOccupancy(f'{self.edge_id}_2')
        # 	# Count total waiting time of the cars in the car lane
        # 	self._total_waiting_time_car += self.traci.lane.getWaitingTime(f'{self.edge_id}_2')
        # 	# Count total number of cars waiting in the car lane
        # 	self._total_count_waiting_car += self.traci.lane.getLastStepHaltingNumber(f'{self.edge_id}_2')
        # 	# Count total number of bikes waiting in the bike lane
        # 	self._total_count_waiting_bike += self.traci.lane.getLastStepHaltingNumber(f'{self.edge_id}_1')
        # 	# Count total number of peds waiting in the ped lane
        # 	self._total_count_waiting_ped += self.traci.lane.getLastStepHaltingNumber(f'{self.edge_id}_0')


        # 	carCollisionCount, bikeCollisionCount, pedCollisionCount = self.getAllCollisionCount()
            
            
        # 	if cosharing:
        # 		#Agent 1
        # 		# Count total number of unique pedestrian on the ped lane
        # 		# self._total_unique_bike_count += 0 # because this lane width is merged into pedestrian
        # 		# Count total number of unique pedestrian + bike on the ped lane
        # 		# self._total_unique_ped_count += self.getUniquePedCount()
        # 		self._unique_ped_count_list.append(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_0'))
        # 		self._unique_bike_count_list.append(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_1'))
        # 		self._total_occupancy_bike_Lane += 0
        # 		# Count total occupancy of ped lane in percentage
        # 		self._total_occupancy_ped_Lane += self.traci.lane.getLastStepOccupancy(f'{self.edge_id}_0')

        # 		#Agent 2
        # 		self._collision_count_bike += bikeCollisionCount
        # 		self._collision_count_ped += pedCollisionCount
                

        # 	else:
        # 		#Agent 1
        # 		self._unique_bike_count_list.append(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_1'))
        # 		self._unique_ped_count_list.append(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_0'))
        # 		# Count total occupancy of bike lane in percentage
        # 		self._total_occupancy_bike_Lane += self.traci.lane.getLastStepOccupancy(f'{self.edge_id}_1')
        # 		# Count total occupancy of ped lane in percentage
        # 		self._total_occupancy_ped_Lane += self.traci.lane.getLastStepOccupancy(f'{self.edge_id}_0')

        # 		#Agent 2
        # 		self._collision_count_bike += bikeCollisionCount
        # 		self._collision_count_ped += pedCollisionCount

        # 	# # # if rewardFlag == True and (self._sumo_step-2)%90 == 0:
        # 	# self._averageRewardStepCounter = self._averageRewardStepCounter + 1
        # 	# queue_length, queue_Count = self.getQueueLength(f'{self.edge_id}_2')
        # 	# self._queue_Length_car_agent_0 += queue_length
        # 	# self._queue_Count_car_agent_0 += queue_Count
        # 	# queue_length,queue_Count = self.getQueueLength(f'{self.edge_id}_0')
        # 	# self._queue_Length_ped_agent_1 += queue_length
        # 	# self._queue_Count_ped_agent_1 += queue_Count
        # 	# queue_length,queue_Count = self.getQueueLength(f'{self.edge_id}_1')
        # 	# self._queue_Length_bike_agent_1 += queue_length
        # 	# self._queue_Count_bike_agent_1 += queue_Count

            
            
        # 	# bikes = self.traci.lane.getLastStepVehicleNumber(f'{self.edge_id}_1') # Number of vehicles on that lane 
        # 	# self._total_bike_on_lane_agent_1 += bikes
        # 	# peds = self.traci.lane.getLastStepVehicleNumber(f'{self.edge_id}_0') # Number of vehicles on that lane 
        # 	# self._total_ped_on_lane_agent_1 += peds
        # 	# perceptionOfSelfDensity = 0.5
        # 	# perceptionOfGroupDensity = 5
        # 	# if 'bicycle' in laneVehicleAllowedType:   
        # 	# 	self._density += perceptionOfGroupDensity*((peds + bikes)/ ((bikeLaneWidth + pedlLaneWidth)*100))
        # 	# else:
        # 	# 	if bikeLaneWidth == 0:
        # 	# 		bikeLaneWidth = 0.01
        # 	# 	if pedlLaneWidth == 0:
        # 	# 		pedlLaneWidth = 0.01
        # 	# 	self._density += perceptionOfSelfDensity*((bikes/(bikeLaneWidth*100) + peds/(pedlLaneWidth*100))/2)

        # self._total_unique_car_count = len(np.unique(np.array(self._unique_car_count_list)))
        # self._total_unique_bike_count = len(np.unique(np.array(self._unique_bike_count_list)))
        # self._total_unique_ped_count = len(np.unique(np.array(self._unique_ped_count_list)))
        # # print("car count =" + str(self._total_vehicle_passed))
        # # print("bike count =" + str(self._total_bike_passed))
        # # print("pedestrian count =" + str(self._total_pedestrian_passed))
        
        for agent in self.agents:
            # obs_n.append(self._get_obs(agent))	
            obs_n[agent.name] = self._get_obs(agent)
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))
            # print("OBSERVATION", self._get_obs(agent), agent.name, agent.edge_id)
        # # all agents get total reward in cooperative case
        # # if self._fatalErroFlag == True:
        # # 	reward = 0
        # self._carQueueLength += self._queue_Length_car_agent_0 / self._averageRewardStepCounter
        # self._bikeQueueLength += self._queue_Length_bike_agent_1 / self._averageRewardStepCounter
        # self._pedQueueLength += self._queue_Length_ped_agent_1 / self._averageRewardStepCounter
        # else:
        self._currentReward = reward_n
        # cooperative_reward = reward_n[0]+reward_n[1]
        # reward_n[0] = cooperative_reward
        # reward_n[1] = cooperative_reward
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] *self.n
        print("Reward = " + str(reward_n))
        self._lastReward = reward_n[0]
        
        # print("reward: " + str(self._lastReward))
        # print("Number of cars passed: " + str(self._total_vehicle_passed))
        return obs_n, reward_n, done_n, info_n


    def rewardAnalysisStats(self):			
        # return self._currentReward[0],self._currentReward[1]
        return self._currentReward[0],self._currentReward[1],self._currentReward[2]

    # set env action for a particular agent
    def _set_action(self, actionDict, modeltype='model', time=None):
        # process action
        if modeltype == "heuristic":
            actions_dict = {}
            for agent in self.edge_agents:
                car_length = 5
                ped_length = 0.215
                bike_length = 1.6
                totalEdgeWidth = 12.6
                # carflow = max(0.01,agent._total_occupancy_car_Lane/car_length)
                # pedflow = max(0.01,agent._total_occupancy_ped_Lane/ped_length)
                # bikeflow = max(0.01,agent._total_occupancy_bike_Lane/bike_length)
                carflow = max(0.01,agent._total_unique_car_count/2)
                pedflow = max(0.01,agent._total_unique_ped_count)
                bikeflow = max(0.01,agent._total_unique_bike_count)
                
                all_flows = [carflow, pedflow, bikeflow]

                alpha = np.clip(carflow/sum(all_flows), 0.1,0.9)
                carLaneWidth = min(max(3.2, alpha*totalEdgeWidth), 9.6)
                alpha = carLaneWidth/totalEdgeWidth

                remainderRoad_0 = totalEdgeWidth - carLaneWidth
                beta = np.clip(bikeflow/sum(all_flows[1:]), 0.1,0.9)
                bikeLaneWidth = max(1.5, beta*remainderRoad_0)
                beta = bikeLaneWidth/remainderRoad_0

                densityThreshold = 1
                if (agent._total_density_ped_lane + agent._total_density_bike_lane) > 2*densityThreshold:
                    coshare = 0
                else:
                    coshare = 1

                actions_dict[("agent 0", agent.edge_id)] = carLaneWidth
                actions_dict[("agent 1", agent.edge_id)] = bikeLaneWidth
                actions_dict[("agent 2", agent.edge_id)] = coshare
            adaptNetwork(self, self.edges,self.base_netfile,actions_dict,modeltype,self._routeFileName,self.sumoCMD, self.pid, self.traci)
        else:
            adaptNetwork(self, self.edges,self.base_netfile,actionDict,modeltype,self._routeFileName,self.sumoCMD, self.pid, self.traci)
            

    def QueueLength(self):
        return self._carQueueLength, self._bikeQueueLength, self._pedQueueLength

    def initSimulator(self,withGUI,portnum):
        if withGUI:
            import traci
        else:
            try:
                import libsumo as traci
            except:
                import traci
        """
        Configure various parameters of SUMO
        """
        # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
            print(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        # self.sumoCMD = ["--time-to-teleport", str(-1),"--person-device.rerouting.probability","1","--person-device.rerouting.period","1","--device.rerouting.probability","1","--device.rerouting.period","1","--ignore-route-errors",
        # 				"--pedestrian.striping.dawdling","0.5","--collision.check-junctions", str(True),"--pedestrian.model","nonInteracting",
        # 				 "--random","-W","--default.carfollowmodel", "IDM","--no-step-log"]
                        # "--device.rerouting.probability","1","--device.rerouting.period","1",
        self.sumoCMD = ["--time-to-teleport.disconnected",str(1),"--ignore-route-errors",
                        "--pedestrian.striping.dawdling","0.5","--collision.check-junctions", "--collision.mingap-factor","0","--collision.action", "warn",
                         "--seed", f"{self.sumo_seed}", "-W","--default.carfollowmodel", "IDM","--no-step-log"]
        if withGUI:
            sumoBinary = checkBinary('sumo-gui')
            self.sumoCMD += ["--start", "--quit-on-end"]
        else:
            sumoBinary = checkBinary('sumo')


        sumoConfig = "gym_sumo/envs/sumo_configs/intersection.sumocfg"
        sumoStartArgs = ['-n', 'gym_sumo/envs/sumo_configs/intersection.net.xml', 
                           '-r', 'gym_sumo/envs/sumo_configs/intersection.rou.xml']
        self.sumoCMD = ["-c", sumoConfig] + self.sumoCMD

        # Initialize the simulation
        traci.start([sumoBinary] + self.sumoCMD + sumoStartArgs)
        return traci

    def set_sumo_seed(self, seed):
        self.sumo_seed = seed

    def nextTimeSlot(self):
        if self._scenario=='Test':
            self._slotId = self.timeOfHour
            self._routeFileName = "testcase_0/two/intersection_Slot_" + str(self._slotId) + ".rou.xml"
        self.timeOfHour +=1

        if self._scenario=='Train':
            self._slotId = np.random.randint(1,120)
            self._routeFileName = "environment/intersection_Slot_" + str(self._slotId) + ".rou.xml"
           
    
    def warmup(self):
        # self._sumo_step = 0
        self.traci.simulationStep(300)
            # self.collectObservation()
            # self._sumo_step +=1
        self._sumo_step = 0
        # for edge_agent in self.edge_agents:
        #     edge_agent.resetAllVariables()

    def _close(self):
        for file in self.generatedFiles:
            os.remove(file)
        self.traci.close()
    
    
def wrapPi(angle):
    # makes a number -pi to pi
        while angle <= -180:
            angle += 360
        while angle > 180:
            angle -= 360
        return angle