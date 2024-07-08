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
from scipy.spatial.distance import cdist


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


        if "agent 1" in agent_name: # bike
            state_0 = laneWidthBike
            state_1 = laneWidthPed				
            state_2 = self.edge_agent._total_occupancy_bike_Lane		
            state_3 = self.edge_agent._total_occupancy_ped_Lane

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
                # self.done = True
            else:
                #occupancy reward. Lower Occupancy higher reward
                reward_occupancy_car = self.edge_agent._total_density_car_lane/10
                # reward_car_Stopped_count = self._total_count_waiting_car/(self.action_steps*10)
                # print("car stopped: " + str(reward_car_Stopped_count))
                reward = -(reward_occupancy_car)*1.5
            # print("agent 0 reward: " + str(reward))
            

        elif "agent 1" in agent_name:
            bikeLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_1')
            pedLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_0')

            if cosharing == True:
                if (bikeLaneWidth + pedLaneWidth) < 2:
                    reward = self.env._fatalPenalty
                    # self.done = True
                else:
                    reward = self.edge_agent._total_occupancy_ped_Lane*10/(self.env.action_steps) # as ped lane will count both waiting bikes and peds since the ped lane is coshared and bike lane width = 0
                    # print("bike + ped stopped in cosharing: " + str(reward))
                    reward = -reward*10
            else:
                if bikeLaneWidth < 1 or pedLaneWidth < 1:
                    reward = self.env._fatalPenalty
                    # self.done = True
                else:
                    # reward = self.edge_agent._total_count_waiting_ped/(self.env.action_steps*10) + self.edge_agent._total_count_waiting_bike/(self.env.action_steps*10)
                    reward_occupancy_bike = self.edge_agent._total_occupancy_bike_Lane/self.env.action_steps
                    reward_occupancy_ped = self.edge_agent._total_occupancy_ped_Lane/self.env.action_steps
                    # print("bike + ped stopped: " + str(reward))
                    reward = -((reward_occupancy_bike+reward_occupancy_ped)/2)*100
                    reward = reward
            # print("agent 1 reward: " + str(reward))
        
        elif "agent 2" in agent_name:
            if cosharing:
                if self.edge_agent._total_density_ped_lane > self.env.density_threshold:
                    reward = -0.75
                elif self.edge_agent._total_density_ped_lane < self.env.density_threshold:
                    reward = 0.75
            else:
                if (self.edge_agent._total_density_ped_lane + self.edge_agent._total_density_bike_lane) > 2*self.env.density_threshold:
                    reward = 0.75
                elif (self.edge_agent._total_density_ped_lane + self.edge_agent._total_density_bike_lane) < 2*self.env.density_threshold:
                    reward = -0.75
            self.edge_agent.reward_agent_2 = reward

            # print("Agent 2 Reward :", self.edge_agent.reward_agent_2)

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
        self._queue_Length_car = 0
        self._queue_Length_ped = 0
        self._queue_Length_bike = 0
        self._queue_Count_car = 0
        self._queue_Count_ped = 0
        self._queue_Count_bike = 0
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
        self._total_hinderance_car_car = 0
        self._total_hinderance_bike_ped = 0
        self._total_hinderance_ped_ped = 0
        self._total_hinderance_car_ped = 0
        self._total_hinderance_car_bike = 0
        self._total_col_car_ped = 0
        self._total_col_car_bike = 0
        self._total_col_car_car = 0
        self._emergencyStoppingVehicleCount = 0
        self._collidingVehicleCount = 0
        self._teleportingVehicleCount = 0

        self._levelOfService = 0 
        
    def collectObservation(self):
        
        if self.edge_id not in self.env.edges: # observations for rest of network
            veh_queue_Count = 0
            veh_waiting_time = 0
            numberLanes = self.traci.edge.getLaneNumber(self.edge_id)
            for n in range(numberLanes):
                lane_id = f'{self.edge_id}_{n}'
                veh_queue_Count += self.env.getLaneQueueLength(lane_id)
                veh_waiting_time += self.env.get_waiting_time_lane(lane_id)
            
            self._queue_Count_car += veh_queue_Count/numberLanes
            self._total_waiting_time_car += veh_waiting_time/numberLanes

            self._total_waiting_time_bike += np.nan
            self._total_waiting_time_ped += np.nan


            self._queue_Count_ped += np.nan
            self._queue_Count_bike += np.nan
            return

        # proceed for main edges
        laneWidthCar = self.traci.lane.getWidth(f'{self.edge_id}_2')
        laneWidthBike = self.traci.lane.getWidth(f'{self.edge_id}_1')
        laneWidthPed = self.traci.lane.getWidth(f'{self.edge_id}_0')
    

        laneVehicleAllowedType = self.traci.lane.getAllowed(f'{self.edge_id}_0')
        if 'bicycle' in laneVehicleAllowedType:
            cosharing = True
        else:
            cosharing = False

        # record observations for each agent
        # Agent 0
        # Count total number of unique cars on the car lane
        self._unique_car_count_list.extend(self.traci.lane.getLastStepVehicleIDs(f'{self.edge_id}_2'))
        # Count total occupancy of car lane in percentage
        self._total_occupancy_car_Lane += self.traci.lane.getLastStepOccupancy(f'{self.edge_id}_2')/laneWidthCar
        # Count total number of cars waiting in the car lane
        self._total_count_waiting_car += self.traci.lane.getLastStepHaltingNumber(f'{self.edge_id}_2')

        [(ped_queue_length, ped_queue_Count), (bike_queue_length, bike_queue_Count),
         (veh_queue_length, veh_queue_Count)] = self.env.getAllQueueLengths(self.edge_id)
        if cosharing:
            ped_queue_length = max(ped_queue_length, bike_queue_length)
            bike_queue_length = ped_queue_length
        self._queue_Length_ped += ped_queue_length
        self._queue_Length_bike += bike_queue_length
        self._queue_Length_car += veh_queue_length
        self._queue_Count_ped += ped_queue_Count
        self._queue_Count_bike += bike_queue_Count
        self._queue_Count_car += veh_queue_Count


        waiting_time_dict = self.env.get_waiting_times(self.edge_id)
        self._total_waiting_time_car += waiting_time_dict['passenger']['wait']
        self._total_waiting_time_bike += waiting_time_dict['bicycle']['wait'] 
        self._total_waiting_time_ped += waiting_time_dict['pedestrian']['wait'] 


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
        if self.env._sumo_step % 10 == 0 and ("Test" in self.env._scenario):
            self._emergencyStoppingVehicleCount += self.traci.simulation.getEmergencyStoppingVehiclesNumber()
            self._collidingVehicleCount += len(self.traci.simulation.getCollidingVehiclesIDList())
            # self._collisions = len(self.traci.simulation.getCollisions())
            self._teleportingVehicleCount += self.traci.simulation.getEndingTeleportNumber()

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
                h_b_b, h_b_p, h_p_p, h_c_p, h_c_b =  self.getHinderanceWhenCosharing(f'{self.edge_id}_0')
                self._total_hinderance_bike_bike += h_b_b
                self._total_hinderance_bike_ped += h_b_p
                self._total_hinderance_ped_ped += h_p_p
                self._total_hinderance_car_ped += h_c_p
                self._total_hinderance_car_bike += h_c_b
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
                self._total_hinderance_bike_bike += self.getHinderance(f'{self.edge_id}_1',"bike_bike")
                self._total_hinderance_ped_ped += self.getHinderance(f'{self.edge_id}_0',"ped_ped")
                self._total_hinderance_car_car += self.getHinderance(f'{self.edge_id}_2',"car_car")

            if self._total_hinderance_car_car > 0:
                t = 0

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
        _occupany_ped_lane = self._total_occupancy_ped_Lane/300
        _occupancy_bike_lane = self._total_occupancy_bike_Lane/300

        _avg_waiting_time_car = self._total_waiting_time_car/(100*300)

        if coSharing:
            self.w_hinderance_b_b = 0.1
            self.w_hinderance_b_p = 0.2
            self.w_hinderance_p_p = 0.1
            laneID = f'{self.edge_id}_0'
            laneWidth = self.traci.lane.getWidth(laneID)#/12.6
            
            los = (_occupany_ped_lane  + _total_hinderance_bike_bike + 
                   _total_hinderance_bike_ped + _total_hinderance_ped_ped)/laneWidth
        else:
            self.w_hinderance_b_b = 0.2
            self.w_hinderance_b_p = 0
            self.w_hinderance_p_p = 0.2
            pedLaneID = f'{self.edge_id}_0'
            bikeLaneID = f'{self.edge_id}_1'
            pedLaneWidth = self.traci.lane.getWidth(pedLaneID)#/12.6
            bikeLaneWidth = self.traci.lane.getWidth(bikeLaneID)#/12.6

            los_ped_Lane =  (_occupany_ped_lane  + _total_hinderance_ped_ped)/pedLaneWidth
            los_bike_Lane = (_occupancy_bike_lane  + _total_hinderance_bike_bike)/bikeLaneWidth
            los = (pedLaneWidth*los_ped_Lane + bikeLaneWidth*los_bike_Lane)/(pedLaneWidth + bikeLaneWidth)

        total_los = los
        return total_los

    def getTestStats(self):
        #returns average waiting time for car, cycle, ped
        #returns average queue length for car, cycle, ped
        #returns LOS per step

        avg_waiting_time_car = self._total_waiting_time_car/self.env.action_steps
        avg_queue_length_car = self._queue_Length_car/self.env.action_steps
        avg_queue_count_car = self._queue_Count_car/self.env.action_steps

        avg_waiting_time_bike = self._total_waiting_time_bike/self.env.action_steps
        avg_queue_length_bike = self._queue_Length_bike/self.env.action_steps
        avg_queue_count_bike = self._queue_Count_bike/self.env.action_steps

        avg_waiting_time_ped = self._total_waiting_time_ped/self.env.action_steps
        avg_queue_length_ped = self._queue_Length_ped/self.env.action_steps
        avg_queue_count_ped = self._queue_Count_ped/self.env.action_steps
        los = self._levelOfService
        safety = self._emergencyStoppingVehicleCount+self._collidingVehicleCount
        teleport = self._teleportingVehicleCount
        laneVehicleAllowedType = self.traci.lane.getAllowed(f'{self.edge_id}_0')
        if 'bicycle' in laneVehicleAllowedType:
            cosharing = True
        else:
            cosharing = False

       
        if len(self.env.edges)==5:
            headers = ['avg_waiting_time_car', 'avg_waiting_time_bike', 'avg_waiting_time_ped',
                    'avg_queue_count_car', 'avg_queue_count_bike', 'avg_queue_count_ped','edge_id']
            values = [avg_waiting_time_car, avg_waiting_time_bike, avg_waiting_time_ped,
                    avg_queue_count_car, avg_queue_count_bike, avg_queue_count_ped,self.edge_id]
        else:
            laneWidth = self.traci.lane.getWidth(f'{self.edge_id}_2')#/12.6
            bikeLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_1')#/12.6
            pedLaneWidth = self.traci.lane.getWidth(f'{self.edge_id}_0')#/12.6

            if len(self.env.edges)!=1:
                headers = ['avg_waiting_time_car', 'avg_waiting_time_bike', 'avg_waiting_time_ped',
                        'avg_queue_count_car', 'avg_queue_count_bike', 'avg_queue_count_ped',
                        'car_lane_width', 'bike_lane_width', 'ped_lane_width',
                        'los', "Reward_Agent_2", "cosharing", 'edge_id','safety','teleport']
                values = [avg_waiting_time_car, avg_waiting_time_bike, avg_waiting_time_ped,
                        avg_queue_count_car, avg_queue_count_bike, avg_queue_count_ped,
                        laneWidth, bikeLaneWidth, pedLaneWidth,
                        los, self.reward_agent_2, cosharing, self.edge_id,safety,teleport]
            else:
                headers = ['avg_waiting_time_car', 'avg_waiting_time_bike', 'avg_waiting_time_ped',
                        'avg_queue_count_car', 'avg_queue_count_bike', 'avg_queue_count_ped',
                        'car_lane_width', 'bike_lane_width', 'ped_lane_width',
                        'los', "Reward_Agent_2", "cosharing", 'ped_safety_counter','bike_safety_counter','veh_safety_counter','edge_id','safety','teleport']
                values = [avg_waiting_time_car, avg_waiting_time_bike, avg_waiting_time_ped,
                        avg_queue_count_car, avg_queue_count_bike, avg_queue_count_ped,
                        laneWidth, bikeLaneWidth, pedLaneWidth,
                        los, self.reward_agent_2, cosharing, self.env.pedSafetyCounter,self.env.bikeSafetyCounter,self.env.vehSafetyCounter,self.edge_id,safety,teleport]
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
                       self._total_occupancy_car_Lane, self._total_occupancy_bike_Lane, self._total_occupancy_ped_Lane, 
                       self._total_density_bike_lane, self._total_density_ped_lane, self._total_density_car_lane,
                       self._levelOfService]

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

    def getHinderanceWhenCosharing(self,laneID):
        bikeList = []
        pedList = []
        carList = []
        
        allVehicles = self.traci.lane.getLastStepVehicleIDs(laneID)         
        if len(allVehicles) > 1:
            for veh in allVehicles:
                x = veh.rsplit("_",1)
                vehID = x[1].split(".",1)
                if vehID[0]=="1":
                    bikeList.append(veh)
                elif vehID[0]=="0":
                    pedList.append(veh)
                elif vehID[0]=="2":
                    carList.append(veh)

        pos_bikes = np.array([list(self.traci.vehicle.getPosition(bike)) for bike in bikeList]).reshape((-1,2))
        pos_peds = np.array([list(self.traci.vehicle.getPosition(ped)) for ped in pedList]).reshape((-1,2))
        pos_cars = np.array([list(self.traci.vehicle.getPosition(car)) for car in carList]).reshape((-1,2))

        pp = cdist(pos_peds,pos_peds)
        bb = cdist(pos_bikes,pos_bikes)
        cc = cdist(pos_cars,pos_cars)
        ##  divide by 2 when doublecounting
        h_p_p = np.sum((0<pp) & (pp<1))/2 # diagonals are self loops
        h_b_b = np.sum((0<bb) & (bb<1))/2 # diagonals are self loops
        h_c_c = np.sum((0<cc) & (cc<1))/2 # diagonals are self loops
        h_b_p = np.sum(cdist(pos_bikes,pos_peds)<1)
        h_c_p = np.sum(cdist(pos_cars,pos_peds)<1)
        h_c_b = np.sum(cdist(pos_cars,pos_bikes)<1)
        return h_b_b,h_b_p,h_p_p,h_c_p,h_c_b

    def getHinderance(self,laneID,betweenVehicleType):
        bikeList = []
        pedList = []
        carList = []
        allVehicles = self.traci.lane.getLastStepVehicleIDs(laneID)
        if len(allVehicles) > 1:
            for veh in allVehicles:
                x = veh.rsplit("_",1)
                vehID = x[1].split(".",1)
                if vehID[0]=="1":
                    bikeList.append(veh)
                elif vehID[0]=="0":
                    pedList.append(veh)
                elif vehID[0]=="2":
                    carList.append(veh)
        pos_bikes = np.array([list(self.traci.vehicle.getPosition(bike)) for bike in bikeList]).reshape((-1,2))
        pos_peds = np.array([list(self.traci.vehicle.getPosition(ped)) for ped in pedList]).reshape((-1,2))
        pos_cars = np.array([list(self.traci.vehicle.getPosition(car)) for car in carList]).reshape((-1,2))

        pp = cdist(pos_peds,pos_peds)
        bb = cdist(pos_bikes,pos_bikes)
        bp = cdist(pos_bikes,pos_peds)
        cc = cdist(pos_cars,pos_cars)

        if betweenVehicleType == "bike_bike":
            hinderance = np.sum((0<bb) & (bb<1))/2 # diagonals are self loops

        elif betweenVehicleType == "bike_ped":
            hinderance = np.sum(bp<1) # no doublecounts
    
        elif betweenVehicleType == "ped_ped":
            hinderance = np.sum((0<pp) & (pp<1))/2 # diagonals are self loops
        
        elif betweenVehicleType == "car_car":
            hinderance = np.sum((0<cc) & (cc<1))/2 # diagonals are self loops

        return hinderance
    
class SUMOEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
    # small_lane_ids = ['E0_2','E0_1','E0_0']

    def __init__(self,reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True,mode='gui',
                 edges=['E0', '-E1','-E2', '-E3'], simulation_end=36000,
                 joint_agents=False, density_threshold=4.87, load_state=False):
        self.pid = os.getpid()
        self.load_state = load_state
        # self.sumoCMD = []
        self.density_threshold = density_threshold
        self.modeltype = 'model'
        self.joint_agents = joint_agents
        self.generatedFiles = []
        self._simulation_end = simulation_end
        self._mode = mode
        np.random.seed(42)
        self.sumo_seed = np.random.randint(69142)
        self.counter = 2
        self.edges = edges
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
        if len(self.edges)==5:
            self.base_netfile = "environment/Barcelona/Barcelona.net.xml"
            self._routeFileName = "environment/Barcelona/Barcelona.rou.xml" # default name 
        else:
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
        
        self._allEdgeNetwork = self.traci.edge.getIDList()
        self._allEdges = [EdgeAgent(self, edge_id) for edge_id in self._allEdgeNetwork if edge_id not in self.edges] # excludes self.edges
        self._num_lane_agents = 3
        
        # configure spaces
        self.edge_agents = [EdgeAgent(self, edge_id) for edge_id in self.edges]
        self._allEdges += self.edge_agents

        if self.joint_agents:
            num_agent_factor = len(self.edge_agents)
        else:
            num_agent_factor = 1
        self.n = self._num_lane_agents*num_agent_factor
        self.agents = self.createNAgents(self.edge_agents)
        self._num_observation = [len(Agent(self, i, self.edge_agents[j]).getState()) for j in range(num_agent_factor) for i in range(self._num_lane_agents)]
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
        self.action_space = spaces.Tuple(self.action_space)
        self.observation_space = spaces.Dict(spaces={agent.name: o_space 
                                                     for i, (agent, o_space) in enumerate(zip(self.agents, self.observation_space))})

    def set_run_mode(self, mode, surge=False): 
        if mode in ['none', 'Test']:
            self._scenario = "Test"
        elif mode == 'Test Single Flow':
            self._scenario = "Test Single Flow"
        else:
            self._scenario = "Train"
        self.is_surge = surge

    def createNAgents(self, edge_agents):

        agents = []
        for j, edge_agent in enumerate(edge_agents):
            edge_id = edge_agent.edge_id
            # edge_agents.append(edge_agent)
            for agent_id in range(0,self._num_lane_agents): #fix this number 3
                agents.append(Agent(self, agent_id, edge_agent))
        return agents
    
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

    def get_waiting_time_lane(self,laneID):
        vehicles = self.traci.lane.getLastStepVehicleIDs(laneID)
        wait_time = 0
        for vehID in vehicles:
            wait_time += self.traci.vehicle.getWaitingTime(vehID)
        return wait_time

    def get_waiting_times(self, edgeID):
        counters = {'bicycle': {'count': 0, 'wait': 0},
                    'passenger': {'count': 0, 'wait': 0},
                    'pedestrian': {'count': 0, 'wait': 0}}
        vehicles = self.traci.edge.getLastStepVehicleIDs(edgeID)
        for vehID in vehicles:
            veh_class = self.traci.vehicle.getVehicleClass(vehID)
            wait_time = self.traci.vehicle.getWaitingTime(vehID)
            counters[veh_class]['count'] += 1
            counters[veh_class]['wait'] += wait_time
        return counters

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
        if len(self.edges)==5:
            temp_agents = self._allEdges
        else:
            temp_agents = self.edge_agents
        for edge_agent in temp_agents:
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
            self._routeFileName = "environment/newTrainFiles/intersection_Slot_" + str(self._slotId) + ".rou.xml"
        elif self._scenario=="Test":
            self._slotId = self.timeOfHour
            if self.is_surge:
                folder = 'two'
            else:
                folder = 'one'
            if len(self.edges)==1:
                print("Testing RESET WAS CALLED", self.timeOfHour)
                self._routeFileName = f"barcelona_test/single/{folder}/intersection_Slot_{self._slotId}.rou.xml"
            elif len(self.edges)==4:
                print("Testing 4wayflow")
                self._routeFileName = f"barcelona_test/4way/{folder}/intersection_Slot_{self._slotId}.rou.xml"            
            elif len(self.edges)==5:
                print("Testing Large Traffic Network")
                self._routeFileName = f"environment/Barcelona/intersection_Slot_{self._slotId}.rou.xml" 
            self.timeOfHour +=1
        elif self._scenario=="Test Single Flow":
            self.timeOfHour = 1
            if len(self.edges)==1:
                print("Testing")
                self._routeFileName = "testcase_0/daytest/flows.rou.xml"
            elif len(self.edges)==4:
                print("Testing 4wayflow")
                self._routeFileName = "testcase_0/4way/flows.rou.xml"
        else:
            self._slotId = np.random.randint(1, 288)
            self._routeFileName = "testcase_1/intersection_Slot_" + str(self._slotId) + ".rou.xml"
        print("Resetting:", self._routeFileName, self.pid, self.sumo_seed)
        
        obs_n = {}
        # self.traci.load(['-n', 'environment/intersection.net.xml', '-r', self._routeFileName, "--start"]) # should we keep the previous vehicle
        if len(self.edges)==5:
            netfile =  'environment/Barcelona/Barcelona.net.xml'
        else:
            netfile = 'environment/intersection.net.xml'
        if self.firstTimeFlag:
            self.traci.load(self.sumoCMD + ['-n', netfile, '-r', self._routeFileName])
            # if self._scenario=="Train":
            while self._sumo_step <= self.action_steps: # THIS IS A WARMUP
                self.traci.simulationStep() 		# Take a simulation step to initialize
                self.collectObservation()
                self._sumo_step +=1
            if self.load_state and self._scenario!="Train":
                self.firstTimeFlag = False
        else:
            print("loading last action")
            if self.modeltype != 'static':
                netfile = f'environment/intersection2_{self.pid}.net.xml'
                if netfile not in self.generatedFiles:
                    self.generatedFiles.append(netfile)
            
            self.traci.load(self.sumoCMD + ['-n', netfile, '-r', self._routeFileName])
            # if self.load_state:
            #     self.traci.simulation.loadState(self.state_file)        
        
        
                
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

    def getLaneQueueLength(self,laneID):
        allVehicles = self.traci.lane.getLastStepVehicleIDs(laneID)
        queueCount = 0
        if len(allVehicles) > 1:
                for veh in allVehicles:
                    speed = self.traci.vehicle.getSpeed(veh)
                    if speed < 0.1:
                        queueCount += 1
        return queueCount

    def getAllQueueLengths(self, edgeID):
        allVehicles = self.traci.edge.getLastStepVehicleIDs(edgeID)
        vehicleDict = {"0": [],
                       "1": [],
                       "2": []}
        for veh in allVehicles:
            x = veh.rsplit('_', 1)
            vehID = x[1].split(".", 1)[0]
            vehicleDict[vehID].append(veh)

        laneLength = self.traci.lane.getLength(f'{edgeID}_0')
        results = []
        for veh_type, vehicles in vehicleDict.items():
            queueCount = 0
            lastQueueLength = 0
            if len(vehicles) > 1:
                for veh in vehicles:
                    speed = self.traci.vehicle.getSpeed(veh)
                    distFromEnd = laneLength - \
                        self.traci.vehicle.getLanePosition(veh)
                    if speed < 0.1:
                        queueCount += 1
                        if distFromEnd > lastQueueLength:
                            lastQueueLength = distFromEnd

                queueLength = lastQueueLength
            else:
                queueLength = 0

            results.append((queueLength, queueCount))
        return results

    def getQueueLength(self, laneID):
        allVehicles = self.traci.lane.getLastStepVehicleIDs(laneID)
        lane_length = self.traci.lane.getLength(laneID)
        queueCount = 0
        queueLength = 0
        if len(allVehicles) > 1:
            for veh in allVehicles:
                speed = self.traci.vehicle.getSpeed(veh)
                distFromEnd = lane_length - self.traci.vehicle.getLanePosition(veh)
                if speed < 0.1:
                    queueCount +=1
                    queueLength = max(distFromEnd, queueLength)
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
        reward = agent.getReward()
        return reward


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
        if len(self.edges)==5:
            #Measure stats for all edges
            for edge_agent in self._allEdges:
                edge_agent.collectObservation()        
        else:
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

        
        #set action for each agents
        if actionFlag == True:
            temp_action_dict = {}
            action_space_dict = {}
            # simple_actions = self.make_action(action_n)
            simple_actions = action_n

            for i, agent in enumerate(self.agents):
                temp_action_dict[(agent.name, agent.edge_id)] = simple_actions[i]

            self._set_action(temp_action_dict, self.modeltype)
            actionFlag = False
            # if 'Test' in self._scenario:
            if not self.load_state:
                self.warmup()
        if self._scenario in ["Train", "Test", "Test Single Flow"]:
            #reset all variables
            if len(self.edges)==5:
                for edge_agent in self._allEdges:
                    edge_agent.resetAllVariables()
                    laneVehicleAllowedType = self.traci.lane.getAllowed(f'{edge_agent.edge_id}_0')
                    if 'bicycle' in laneVehicleAllowedType:
                        edge_agent.cosharing = True
                    else:
                        edge_agent.cosharing = False
            else:
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

        
        for agent in self.agents:
            obs_n[agent.name] = self._get_obs(agent)
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))
        # all agents get total reward in cooperative case
        self._currentReward = reward_n
        # cooperative_reward = reward_n[0]+reward_n[1]
        # reward_n[0] = cooperative_reward
        # reward_n[1] = cooperative_reward
        reward = np.sum(reward_n)
        # print("Reward = " + str(reward_n), done_n)
        if self.shared_reward:
            reward_n = [reward] *self.n
        self._lastReward = reward_n[0]
        
        # print("reward: " + str(self._lastReward))
        # print("Number of cars passed: " + str(self._total_vehicle_passed))
        return obs_n, reward_n, done_n, info_n


    def rewardAnalysisStats(self):			
        return self._currentReward

    @property
    def getAgentNames(self):
        return [agent.name for agent in self.agents]
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
                carflow = max(0.01,agent._total_density_car_lane)
                pedflow = max(0.01,agent._total_density_ped_lane)
                bikeflow = max(0.01,agent._total_density_bike_lane)
                
                all_flows = [carflow, pedflow, bikeflow]

                alpha = np.clip(carflow/sum(all_flows), 0.1,0.9)
                carLaneWidth = min(max(3.2, alpha*totalEdgeWidth), 9.6)
                alpha = carLaneWidth/totalEdgeWidth

                remainderRoad_0 = totalEdgeWidth - carLaneWidth
                beta = np.clip(bikeflow/sum(all_flows[1:]), 0.1,0.9)
                bikeLaneWidth = max(1.5, beta*remainderRoad_0)
                beta = bikeLaneWidth/remainderRoad_0

                if (agent._total_density_ped_lane + agent._total_density_bike_lane) > 2*self.density_threshold:
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
        if self.withGUI:
            import traci
        else:
            try:
                import libsumo as traci
            except:
                import traci
        """
        Configure various parameters of SUMO
        """

        if self.withGUI:
            sumoBinary = checkBinary('sumo-gui')
            # self.sumoCMD += ["--start", "--quit-on-end"]
        else:
            sumoBinary = checkBinary('sumo')

        # sumoConfig = "gym_sumo/envs/sumo_configs/intersection.sumocfg"
        # self.sumoCMD = ["-c", sumoConfig] + self.sumoCMD
        if len(self.edges)==5:
            sumoStartArgs = ['-n', 'environment/Barcelona/Barcelona.net.xml', 
                        '-r', 'environment/Barcelona/Barcelona.rou.xml']
        else:
            sumoStartArgs = ['-n', 'environment/intersection.net.xml', 
                            '-r', 'gym_sumo/envs/sumo_configs/intersection.rou.xml']

        # Initialize the simulation
        traci.start([sumoBinary] + self.sumoCMD + sumoStartArgs)
        return traci

    @property
    def sumoCMD(self):
        sumocmd = ["--time-to-teleport.disconnected",str(5), "--ignore-route-errors","--collision.mingap-factor","0",
                        "--pedestrian.striping.dawdling","0.5","--collision.check-junctions","--collision.action", "warn",
                        "--seed", f"{self.sumo_seed}", "-W","--default.carfollowmodel", "IDM","--no-step-log", "--save-state.transportables"]
        if self.withGUI:
            sumocmd += ["--start", "--quit-on-end"]
        if len(self.edges)==5:
            sumoConfig = "environment/Barcelona/Barcelona.sumocfg"
        else:
            sumoConfig = "gym_sumo/envs/sumo_configs/intersection.sumocfg"
        sumocmd = ["-c", sumoConfig] + sumocmd
        return sumocmd
    
    def seed(self, seed):
        self.sumo_seed = seed
        np.random.random(seed)
        return super().seed(seed)


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