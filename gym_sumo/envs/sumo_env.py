from gym import Env
from gym import spaces
from gym.envs.registration import EnvSpec
from gym.utils import seeding
import traci
# import traci.constants as tc
# import libsumo as traci
# from scipy.misc import imread
from gym import spaces
from string import Template
import numpy as np
import math
import time
import random
from cv2 import imread,imshow,resize
import cv2
from collections import namedtuple
from sumolib import checkBinary
import os, sys, subprocess
from gym_sumo.envs.adapt_network import adaptNetwork
from gym_sumo.envs.adapt_route_file import adaptRouteFile
import xml.etree.ElementTree as ET
import math
from itertools import combinations
from collections import Counter

class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	small_lane_ids = ['E0_2','E0_1','E0_0']

	def __init__(self,reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True,mode='gui',simulation_end=36000):

		self.sumoCMD = []
		self._simulation_end = simulation_end
		self._mode = mode
		self._seed(40)
		self.counter = 2
		self.traci = self.initSimulator(True,8870)
		self._sumo_step = 0
		self._episode = 0
		self._flag = True       
		self._weightCar = 1
		self._weightPed = 10
		self._weightBike = 3
		self._weightBikePed = 2 
		# self._gamma = 0.75
		self._slotId = 1
		self._routeFileName = "environment/intersection_Slot_1.rou.xml" # default name 
		self._max_steps = 24000
		self._slot_duration = 1200
		self._max_slots = 3
		self._num_observation = 6
		self._num_actions = 1
		self._reward_store = []
		self._cumulative_wait_store = []
		self._avg_queue_length_store = []
		self.action_steps = 300
		self.sumo_running = False
		self.viewer = None
		self.firstTimeFlag = True
		self._total_vehicle_passed_agent_0 = 0 
		self._total_pedestrian_passed_agent_1 = 0
		self._total_bike_passed_agent_1 = 0
		self._total_vehicle_on_lane_agent_0 = 0
		self._total_bike_on_lane_agent_1 = 0
		self._total_ped_on_lane_agent_1 = 0
		self._density = 0
		self._avg_ped_distance_agent_1 = 0
		self._avg_bike_distance_agent_1 = 0
		self._queue_Length_car_agent_0 = 0
		self._queue_Length_ped_agent_1 = 0
		self._queue_Length_bike_agent_1 = 0
		self._queue_Count_car_agent_0 = 0
		self._queue_Count_ped_agent_1 = 0
		self._queue_Count_bike_agent_1 = 0
		self._total_vehicle_passed_agent_2 = 0 
		self._unique_car_count_list = []
		self._unique_ped_count_list = []
		self._unique_bike_count_list = []
		self._total_waiting_time_car = 0
		self._total_unique_car_count = 0
		self._total_unique_bike_count = 0
		self._total_unique_ped_count = 0
		self._total_occupancy_car_Lane = 0
		self._total_occupancy_bike_Lane = 0
		self._total_occupancy_ped_Lane = 0
		self._total_count_waiting_car = 0
		self._total_count_waiting_bike = 0
		self._total_count_waiting_ped = 0
		self._collision_count_bike = 0
		self._collision_count_ped = 0

		# set required vectorized gym env property
		self.n = 3
		self.agents = self.createNAgents()
		self._lastReward = 0
		# self.observation = self.reset()

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

		self.lanes_length = 100
		# configure spaces
		self.action_space = []
		self.observation_space = []
		for agent in self.agents:
			self.action_space.append(spaces.Box(low=0, high=+1, shape=(1,))) # alpha value
			# self.action_space.append(spaces.Discrete(self._num_actions))
			# observation space
			self.observation_space.append(spaces.Box(low=0, high=+1, shape=(self._num_observation,)))
			# if agent.name == "agent 0":
			# 	self.observation_space.append(spaces.Box(low=0, high=+1, shape=(self._num_observation,)))
			# else:
			# 	self.observation_space.append(spaces.Box(low=0, high=+1, shape=(self._num_observation+1,)))

		# self.action_space = spaces.Box(low=np.array([0]), high= np.array([+1])) # Beta value 
		# self.observation_space = spaces.Box(low=0, high=1, shape=(np.shape(self.observation)))
		# self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32))

	def createNAgents(self):
		agents = [Agent() for i in range(self.n)]
		for i, agent in enumerate(agents):
			agent.name = 'agent %d' % i

		return agents

		return agents
	
	# Edge E0 - Agent 1 and Agent 2
	# Edge E0 - Agent 1 and Agent 2
	# Edge E0 - Agent 1 and Agent 2

	
	def getState(self,agent):
		"""
		Retrieve the state of the network from sumo. 
		"""
		state = np.zeros(self._num_observation,dtype=np.float32)
		normalizeUniqueVehicleCount = 1000
		laneWidthCar = self.traci.lane.getWidth('E0_2')
		laneWidthBike = self.traci.lane.getWidth('E0_1')
		laneWidthPed = self.traci.lane.getWidth('E0_0')
		nLaneWidthCar = np.interp(laneWidthCar, [0,12.6], [0,1])
		nLaneWidthBike = np.interp(laneWidthBike, [0,12.6], [0,1])
		nLaneWidthPed = np.interp(laneWidthPed, [0,12.6], [0,1])

		#E0 is for agent 0 and 1, #-E0 is for agent 2 and 3, #E1 is for agent 4 and 5, #-E1 is for agent 6 and 7
		#E2 is for agent 8 and 9, #-E2 is for agent 10 and 11, #E3 is for agent 12 and 13, #-E3 is for agent 14 and 15

		laneVehicleAllowedType =traci.lane.getAllowed('E0_0')
		if 'bicycle' in laneVehicleAllowedType:
			cosharing = True
		else:
			cosharing = False

		if agent.name == "agent 0": # car
			state[0] = nLaneWidthCar
			state[1] = nLaneWidthBike + nLaneWidthPed
			state[2] = np.interp(self._total_unique_car_count,[0,normalizeUniqueVehicleCount],[0,1]) # average number of unique cars on the car lane through simulation steps
			state[3] = self._total_occupancy_car_Lane/self.action_steps # average occupancy of the car lane through simulation steps. The raw value is in percentage
			state[4] = np.interp(self._total_waiting_time_car/self.action_steps,[0,100],[0,1]) # average waiting time for cars through simulation steps
			state[5] = np.interp(self._total_count_waiting_car/self.action_steps,[0,10],[0,1]) # average waiting count of cars on the car lane through simulation steps

		
		elif agent.name == "agent 1": # bike
			if cosharing:
				state[0] = nLaneWidthBike
				state[1] = nLaneWidthPed
				state[2]= 0
				state[3] = np.interp(self._total_unique_ped_count,[0,normalizeUniqueVehicleCount],[0,1]) # average number of unique peds + bikes on the bike lane through simulation steps
				state[4] = 0 # average occupancy of the bike lane through simulation steps. The raw value is in percentage
				state[5] = self._total_occupancy_ped_Lane/self.action_steps # average occupancy of the ped lane through simulation steps. The raw value is in percentage
			else:
				state[0] = nLaneWidthBike
				state[1] = nLaneWidthPed
				state[2] = np.interp(self._total_unique_bike_count,[0,normalizeUniqueVehicleCount],[0,1]) # average number of unique bikes on the bike lane through simulation steps
				state[3] = np.interp(self._total_unique_ped_count,[0,normalizeUniqueVehicleCount],[0,1]) # average number of unique peds on the ped lane through simulation steps
				state[4] = self._total_occupancy_bike_Lane/self.action_steps # average occupancy of the bike lane through simulation steps. The raw value is in percentage
				state[5] = self._total_occupancy_ped_Lane/self.action_steps # average occupancy of the ped lane through simulation steps. The raw value is in percentage
		

		elif agent.name == "agent 2": 
			if cosharing:
				avg_ped_count = self._total_unique_ped_count/self.action_steps
				pedDensity = avg_ped_count/(laneWidthBike + laneWidthPed)
				state[0] = 1 #flag for cosharing on or off
				state[1] = pedDensity # pedestrian + bike density
				state[2] = 0
				state[3] = np.interp(self._collision_count_bike/self.action_steps,[0,10],[0,1])  #avg. collision count bike
				state[4] = np.interp(self._collision_count_ped/self.action_steps,[0,10],[0,1]) #avg. collision count ped
				state[5] = 0 #zero padding
			else:
				avg_ped_count = self._total_unique_ped_count/self.action_steps
				pedDensity = avg_ped_count/laneWidthPed

				avg_bike_count = self._total_unique_bike_count/self.action_steps
				bikeDensity = avg_bike_count/laneWidthBike

				state[0] = 0 #flag for cosharing on or off
				state[1] = pedDensity # pedestrian
				state[2] = bikeDensity # bike density
				state[3] = np.interp(self._collision_count_bike/self.action_steps,[0,10],[0,1])  #avg. collision count bike
				state[4] = np.interp(self._collision_count_ped/self.action_steps,[0,10],[0,1]) #avg. collision count ped
				state[5] = 0 #zero padding

		return state
	
	def _collect_waiting_times_cars(self,laneID):
		"""
		Retrieve the waiting time of every car in the incoming roads
		"""

		# car_list = traci.vehicle.getIDList()
		nCars= traci.lane.getLastStepVehicleIDs(laneID)
		waiting_times = {}
		avg_waiting_time = 0
		if len(nCars) > 0:
			for car_id in nCars:
				wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
				waiting_times[car_id] = wait_time
				
			avg_waiting_time = sum(waiting_times.values())/len(nCars)
			temp_total_wait_time = traci.lane.getWaitingTime(laneID)
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

	def getActionStateAfterWarmUpPeriod(self):
		#record observatinos for each agent
		obs_n = []
		for agent in self.agents:
			obs_n.append(self._get_obs(agent))
		return obs_n
	
	def outPutLaneWidth(self):
		for ilane in range(0, 3):
			lane_id = self.small_lane_ids[ilane]
			tempLaneWidth = self.traci.lane.getWidth(lane_id)
			print(lane_id + ":" + str(tempLaneWidth))

	def get_lane_queue(self,lane,min_gap):
		lanes_queue = self.traci.lane.getLastStepHaltingNumber(lane) / (self.lanes_length/ (min_gap + self.traci.lane.getLastStepLength(lane)))
	
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
		self._unique_car_count_list.clear()
		self._unique_ped_count_list.clear()
		self._unique_bike_count_list.clear()
		self._total_unique_car_count = 0
		self._total_unique_bike_count = 0
		self._total_unique_ped_count = 0
		self._total_occupancy_car_Lane = 0
		self._total_waiting_time_car = 0
		self._total_count_waiting_car = 0
		self._total_count_waiting_bike = 0
		self._total_count_waiting_ped = 0
		self._total_occupancy_bike_Lane = 0
		self._total_occupancy_ped_Lane = 0
		self._collision_count_bike = 0
		self._collision_count_ped = 0


	def reset(self,testFlag):		
		self._sumo_step = 0
		self.resetAllVariables()
		
		self._slotId = random.randint(1, 27)
		#Adapt Route File for continous change
		# self._slotId = 24 # temporary
		adaptRouteFile(self._slotId)
		self._routeFileName = "environment/intersection_Slot_" + str(self._slotId) + ".rou.xml"
		print(self._routeFileName)
		# print("Car Flow for E0: " + str(self.readRouteFile("car")))
		# print("Ped Flow for E0: " + str(self.readRouteFile("ped")))
		# print("Bike Flow for E0: " + str(self.readRouteFile("bike")))
		# # save state
		# self.traci.simulation.saveState('environment\savedstate.xml') 
		obs_n = []
		if testFlag == True:
			for agent in self.agents:
						obs_n.append(self._get_obs(agent))
		else:			
			# traci.load(['-n', 'environment/intersection.net.xml', '-r', self._routeFileName, "--start"]) # should we keep the previous vehic
			traci.load(self.sumoCMD + ['-n', 'environment/intersection.net.xml', '-r', self._routeFileName])
			while self._sumo_step <= self.action_steps:
				self.traci.simulationStep() 		# Take a simulation step to initialize
				#compute sum of queue length for all lanes
				queue_length, queueCount = self.getQueueLength('E0_2')
				self._queue_Length_car_agent_0 += queue_length
				queue_length, queueCount = self.getQueueLength('E0_0')
				self._queue_Length_ped_agent_1 += queue_length
				queue_length, queueCount = self.getQueueLength('E0_1')
				self._queue_Length_bike_agent_1 += queue_length
				self.firstTimeFlag = False
				self._sumo_step +=1
			
					
			#record observatinos for each agent
			for agent in self.agents:
				agent.done = False
				obs_n.append(self._get_obs(agent))

		return obs_n

	# get observation for a particular agent
	def _get_obs(self, agent):
		return self.getState(agent)

	# def _observation(self,agent):
	# 	return self.getState(agent)

	def _render(self, mode='gui', close=False):

		if self.mode == "gui":
			img = imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),'sumo.png'), 1)
			if mode == 'rgb_array':
				return img
			elif mode == 'human':
				from gym.envs.classic_control import rendering
				if self.viewer is None:
					self.viewer = rendering.SimpleImageViewer()
				self.viewer.imshow(img)
		else:
			raise NotImplementedError("Only rendering in GUI mode is supported")


	def getQueueLength(self,laneID):
		allVehicles = self.traci.lane.getLastStepVehicleIDs(laneID)
		queueCount = 0
		if len(allVehicles) > 1:
			lastMaxPos = 60
			for veh in allVehicles:
				speed = traci.vehicle.getSpeed(veh)
				pos_p_x = traci.vehicle.getPosition(veh)[0]
				if speed < 0.1 and pos_p_x > -35:					
					# pos_p_y = traci.vehicle.getPosition(veh)[1]
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
				pos_p_x.append(traci.vehicle.getPosition(ped)[0])
				pos_p_y.append(traci.vehicle.getPosition(ped)[1])
			
			points = list(zip(pos_p_x,pos_p_y))
			distances = [math.dist(p1, p2) for p1, p2 in combinations(points, 2)]
			avg_distance_pedestrian = sum(distances) / len(distances)
			return avg_distance_pedestrian
		else:
			return 0

	# function to compute spacing between all bikes 
	def interDistanceBetweenBikes(self,BikeLaneID):
		#loop through all pedestrian on a lane
		allBikes = self.traci.lane.getLastStepVehicleIDs(BikeLaneID)
		if len(allBikes) > 1:
			pos_p_x = []
			pos_p_y = []
			for bike in allBikes:
				pos_p_x.append(traci.vehicle.getPosition(bike)[0])
				pos_p_y.append(traci.vehicle.getPosition(bike)[1])
			
			points = list(zip(pos_p_x,pos_p_y))
			distances = [math.dist(p1, p2) for p1, p2 in combinations(points, 2)]
			avg_distance_bikes = sum(distances) / len(distances)
			return avg_distance_bikes
		else:
			return 0

	# get reward for a particular agent
	def _get_reward(self,agent):
		
		# defaultCarLength = 5
		# defaultPedLength = 0.215
		# defaultBikeLength = 1.6
		laneVehicleAllowedType =traci.lane.getAllowed('E0_0')
		cosharing = False
		if 'bicycle' in laneVehicleAllowedType: 
			cosharing = True
		if agent.name == "agent 0":
			carLaneWidth = self.traci.lane.getWidth('E0_2')
			if carLaneWidth < 3.2:
				reward = -10.0
				agent.done = True
			else:
				reward_car_Stopped_count = self._total_count_waiting_car/self.action_steps
				reward = -reward_car_Stopped_count/10
				
			

		elif agent.name == "agent 1":
			bikeLaneWidth = self.traci.lane.getWidth('E0_1')
			pedLaneWidth = self.traci.lane.getWidth('E0_0')

			if cosharing == True:
				if (bikeLaneWidth + pedLaneWidth) < 2:
					reward = -10.0
					agent.done = True
				else:
					reward = self._total_count_waiting_ped/self.action_steps # as ped lane will count both waiting bikes and peds since the ped lane is coshared and bike lane width = 0
					reward = -reward/100
			else:
				if bikeLaneWidth < 1 or pedLaneWidth < 1:
					reward = -10.0
					agent.done = True
				else:
					reward = self._total_count_waiting_ped/(self.action_steps*100) + self._total_count_waiting_bike/(self.action_steps*100)
					reward = -reward
		
		
		elif agent.name == "agent 2":
			if cosharing == True:
				positive_reward_cosharing = +10
				collisionCount = self._collision_count_bike + self._collision_count_ped
				negative_reward_collision = -0.1*collisionCount
				reward = positive_reward_cosharing + negative_reward_collision
			else:
				positive_reward_cosharing = +0
				reward = positive_reward_cosharing
			# collisionCount = self._collision_count_bike/self.action_steps + self._collision_count_ped/self.action_steps
			

				
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
		carsCountList = self.traci.lane.getLastStepVehicleIDs('E0_2')
		self._unique_car_count_list.append(carsCountList)
		num_values = len(np.unique(np.array(self._unique_car_count_list)))
		return num_values

	def getUniquePedCount(self):
		pedCountList = self.traci.lane.getLastStepVehicleIDs('E0_0')
		self._unique_ped_count_list.append(pedCountList)
		num_values = len(np.unique(np.array(self._unique_ped_count_list)))
		return num_values

	def getUniqueBikeCount(self):
		bikeCountList = self.traci.lane.getLastStepVehicleIDs('E0_1')
		self._unique_bike_count_list.append(bikeCountList)
		num_values = len(np.unique(np.array(self._unique_bike_count_list)))
		return num_values

	def getAllCollisionCount(self):		
		allCollidingVehicleIDList = self.traci.simulation.getCollidingVehiclesIDList()
		bikeCollisionCounter = 0
		pedCollisionCounter = 0
		carCollisionCounter = 0
		for veh in allCollidingVehicleIDList:
			# print(veh)
			x = veh.split("_",2)
			vehID = x[1].split(".",1)
			if vehID[0]=="1":
				bikeCollisionCounter +=1
			elif vehID[0]=="0":
				pedCollisionCounter +=1
			elif vehID[0]=="2":
				carCollisionCounter +=1

		return carCollisionCounter,bikeCollisionCounter,pedCollisionCounter
	
	def _step(self,action_n):
		obs_n = []
		reward_n = []
		done_n = []
		info_n = {'n':[]}
		rewardFlag = False
		actionFlag = True
		for agent in self.agents:
				agent.done = False
		
		self._sumo_step = 0
		#simulating a warm period of N=self.action_steps  and then recording the state, action, reward tuple. 
		bikeLaneWidth = traci.lane.getWidth('E0_1')
		pedlLaneWidth = traci.lane.getWidth('E0_0')
		carLaneWidth = traci.lane.getWidth('E0_2')
		
		#reset all variables
		self.resetAllVariables()
		laneVehicleAllowedType =traci.lane.getAllowed('E0_0')
	
		while self._sumo_step <= self.action_steps:
			# advance world state	
			self.traci.simulationStep()
			self._sumo_step +=1
			#set action for each agents
			if actionFlag == True:
				temp_action_dict = {}
				action_space_dict = {}
				for i, agent in enumerate(self.agents):
					# index = np.argmax(action_n[i])
					# temp_action_dict[agent.name] = int(index)
					#for continous action space
					temp_action_dict[agent.name] = action_n[i]
					action_space_dict[agent.name] = self.action_space[i]
				self._set_action(temp_action_dict,agent,action_space_dict)
				actionFlag = False
			
			if 'bicycle' in laneVehicleAllowedType:
				cosharing = True
			else:
				cosharing = False
			# record observatinos for each agent
			#Agent 0
			# # Count total number of unique cars on the car lane
			self._unique_car_count_list.append(self.traci.lane.getLastStepVehicleIDs('E0_2'))
			# Count total occupancy of car lane in percentage
			self._total_occupancy_car_Lane += self.traci.lane.getLastStepOccupancy('E0_2')
			# Count total waiting time of the cars in the car lane
			self._total_waiting_time_car += self.traci.lane.getWaitingTime('E0_2')
			# Count total number of cars waiting in the car lane
			self._total_count_waiting_car += self.traci.lane.getLastStepHaltingNumber('E0_2')
			# Count total number of bikes waiting in the bike lane
			self._total_count_waiting_bike += self.traci.lane.getLastStepHaltingNumber('E0_1')
			# Count total number of peds waiting in the ped lane
			self._total_count_waiting_ped += self.traci.lane.getLastStepHaltingNumber('E0_0')


			carCollisionCount, bikeCollisionCount, pedCollisionCount = self.getAllCollisionCount()
			
			
			if cosharing:
				#Agent 1
				# Count total number of unique pedestrian on the ped lane
				# self._total_unique_bike_count += 0 # because this lane width is merged into pedestrian
				# Count total number of unique pedestrian + bike on the ped lane
				# self._total_unique_ped_count += self.getUniquePedCount()
				self._unique_ped_count_list.append(self.traci.lane.getLastStepVehicleIDs('E0_0'))
				self._unique_bike_count_list.append(self.traci.lane.getLastStepVehicleIDs('E0_1'))
				self._total_occupancy_bike_Lane += 0
				# Count total occupancy of ped lane in percentage
				self._total_occupancy_ped_Lane += self.traci.lane.getLastStepOccupancy('E0_0')

				#Agent 2
				self._collision_count_bike += bikeCollisionCount
				self._collision_count_ped += pedCollisionCount
				

			else:
				#Agent 1
				self._unique_bike_count_list.append(self.traci.lane.getLastStepVehicleIDs('E0_1'))
				self._unique_ped_count_list.append(self.traci.lane.getLastStepVehicleIDs('E0_0'))
				# Count total occupancy of bike lane in percentage
				self._total_occupancy_bike_Lane += self.traci.lane.getLastStepOccupancy('E0_1')
				# Count total occupancy of ped lane in percentage
				self._total_occupancy_ped_Lane += self.traci.lane.getLastStepOccupancy('E0_0')

				#Agent 2
				self._collision_count_bike += bikeCollisionCount
				self._collision_count_ped += pedCollisionCount

			# # # if rewardFlag == True and (self._sumo_step-2)%90 == 0:
			# self._averageRewardStepCounter = self._averageRewardStepCounter + 1
			# queue_length, queue_Count = self.getQueueLength('E0_2')
			# self._queue_Length_car_agent_0 += queue_length
			# self._queue_Count_car_agent_0 += queue_Count
			# queue_length,queue_Count = self.getQueueLength('E0_0')
			# self._queue_Length_ped_agent_1 += queue_length
			# self._queue_Count_ped_agent_1 += queue_Count
			# queue_length,queue_Count = self.getQueueLength('E0_1')
			# self._queue_Length_bike_agent_1 += queue_length
			# self._queue_Count_bike_agent_1 += queue_Count

			
			
			# bikes = self.traci.lane.getLastStepVehicleNumber('E0_1') # Number of vehicles on that lane 
			# self._total_bike_on_lane_agent_1 += bikes
			# peds = self.traci.lane.getLastStepVehicleNumber('E0_0') # Number of vehicles on that lane 
			# self._total_ped_on_lane_agent_1 += peds
			# perceptionOfSelfDensity = 0.5
			# perceptionOfGroupDensity = 5
			# if 'bicycle' in laneVehicleAllowedType:   
			# 	self._density += perceptionOfGroupDensity*((peds + bikes)/ ((bikeLaneWidth + pedlLaneWidth)*100))
			# else:
			# 	if bikeLaneWidth == 0:
			# 		bikeLaneWidth = 0.01
			# 	if pedlLaneWidth == 0:
			# 		pedlLaneWidth = 0.01
			# 	self._density += perceptionOfSelfDensity*((bikes/(bikeLaneWidth*100) + peds/(pedlLaneWidth*100))/2)

		self._total_unique_car_count = len(np.unique(np.array(self._unique_car_count_list)))
		self._total_unique_bike_count = len(np.unique(np.array(self._unique_bike_count_list)))
		self._total_unique_ped_count = len(np.unique(np.array(self._unique_ped_count_list)))
		# print("car count =" + str(self._total_vehicle_passed))
		# print("bike count =" + str(self._total_bike_passed))
		# print("pedestrian count =" + str(self._total_pedestrian_passed))

		for agent in self.agents:
			obs_n.append(self._get_obs(agent))			
			reward_n.append(self._get_reward(agent))
			print(self._get_reward(agent))
			done_n.append(self._get_done(agent))

			info_n['n'].append(self._get_info(agent))

		# # all agents get total reward in cooperative case
		# # if self._fatalErroFlag == True:
		# # 	reward = 0
		# self._carQueueLength += self._queue_Length_car_agent_0 / self._averageRewardStepCounter
		# self._bikeQueueLength += self._queue_Length_bike_agent_1 / self._averageRewardStepCounter
		# self._pedQueueLength += self._queue_Length_ped_agent_1 / self._averageRewardStepCounter
		# else:
		reward = np.sum(reward_n)
		if self.shared_reward:
			reward_n = [reward] *self.n
		# print("Reward = " + str(reward_n))
		self._lastReward = reward_n[0]
		print("reward: " + str(self._lastReward))
		# print("Number of cars passed: " + str(self._total_vehicle_passed))
		return obs_n, reward_n, done_n, info_n


	# set env action for a particular agent
	def _set_action(self, actionDict, agent, action_space, time=None):
		# process action
		adaptNetwork(actionDict,agent.name,self._routeFileName,self.sumoCMD)
		# t = 0

	def QueueLength(self):
		return self._carQueueLength, self._bikeQueueLength, self._pedQueueLength

	def initSimulator(self,withGUI,portnum):
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
		self.sumoCMD = ["--time-to-teleport.disconnected",str(1),"--ignore-route-errors","--device.rerouting.probability","1","--device.rerouting.period","1",
						"--pedestrian.striping.dawdling","0.5","--collision.check-junctions", str(True),
						 "--random","-W","--default.carfollowmodel", "IDM","--no-step-log", "--statistic-output","output.xml"]
		if withGUI:
			sumoBinary = checkBinary('sumo-gui')
			self.sumoCMD += ["--start"]
		else:
			sumoBinary = checkBinary('sumo')


		sumoConfig = "C:/D/SUMO/MARL/multiagentRL/gym_sumo/envs/sumo_configs/intersection.sumocfg"

		# Call the sumo simulator
		# sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(portnum), \
		# 	"--time-to-teleport", str(-1), "--collision.check-junctions", str(True), \
		# 	"--random","--log", "log.txt","--error-log","errorLog.txt","--default.carfollowmodel", "IDM"])

		

		# sumoProcess = subprocess.Popen([sumoBinary] + self.sumoCMD + ["-c", sumoConfig,"--remote-port", str(portnum)])



		# sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--no-step-log", "true", "--waiting-time-memory", str(3600), "--default.carfollowmodel", "IDM"], stdout=sys.stdout, stderr=sys.stderr)

		# sumoProcess = [sumoBinary, "-c", sumoConfig, "--no-step-log", "true", "--waiting-time-memory", str(3600), "--default.carfollowmodel", "IDM"]

		# Initialize the simulation
		# traci.init(portnum)
		print(" ".join([sumoBinary] + self.sumoCMD + ["-c", sumoConfig]))
		traci.start([sumoBinary] + self.sumoCMD + ["-c", sumoConfig])
		return traci

	def closeSimulator(traci):
		traci.close()
		sys.stdout.flush()
	
	
def wrapPi(angle):
	# makes a number -pi to pi
		while angle <= -180:
			angle += 360
		while angle > 180:
			angle -= 360
		return angle


# properties of agent entities
class Agent():
    def __init__(self):
        super(Agent, self).__init__()
        self.done = False
        # script behavior to execute
        self.action_callback = None