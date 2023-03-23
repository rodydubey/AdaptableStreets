from gym import Env
from gym import spaces
from gym.utils import seeding
from gym import spaces
import numpy as np
import math
from sumolib import checkBinary
import os, sys
sys.path.append('../') #allows loading of agent.py
from agent import Agent
from gym_sumo.envs.adapt_network import adaptNetwork, carLane_width_actions, bikeLane_width_actions
from gym_sumo.envs.adapt_route_file import adaptRouteFile
import xml.etree.ElementTree as ET
import math
from itertools import combinations
from agent import Agent

class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	small_lane_ids = ['E0_2','E0_1','E0_0']

	def __init__(self,reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True,mode='gui',simulation_end=36000):
		self.pid = os.getpid()
		self.sumoCMD = []
		self._simulation_end = simulation_end
		self._mode = mode
		# self._seed(40)
		np.random.seed(42)
		self.counter = 2
		self.withGUI = mode=='gui'
		self.traci = self.initSimulator(self.withGUI, self.pid)
		self._sumo_step = 0
		self._episode = 0
		self._flag = True       
		self._weightCar = 1
		self._weightPed = 10
		self._weightBike = 3
		self._weightBikePed = 2 
		# self._gamma = 0.75
		self._slotId = 1
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
		self.coShareValue = 0
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
		self._total_mean_speed_car = 0
		self._total_mean_speed_bike = 0
		self._total_mean_speed_ped = 0
		self._total_count_waiting_bike = 0
		self._total_count_waiting_ped = 0
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
		self._currentReward = []
		self._lastReward = 0
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

		self._scenario = "Train"
		# set required vectorized gym env property
		self.n = 3
		
		# configure spaces
		self._num_observation = [len(self.getState(f'agent {i}')) for i in range(self.n)]
		self._num_actions = [len(carLane_width_actions), len(bikeLane_width_actions),2]
		self.action_space = []
		self.observation_space = []
		for i in range(self.n):
			if self._num_actions[i]==1:
				self.action_space.append(spaces.Box(low=0, high=+1, shape=(1,))) # alpha value
			else:
				self.action_space.append(spaces.Discrete(self._num_actions[i]))
			# observation space
			self.observation_space.append(spaces.Box(low=0, high=+1, shape=(self._num_observation[i],)))
			# if agent.name == "agent 0":
			# 	self.observation_space.append(spaces.Box(low=0, high=+1, shape=(self._num_observation,)))
			# else:
			# 	self.observation_space.append(spaces.Box(low=0, high=+1, shape=(self._num_observation+1,)))
		self.agents = self.createNAgents()
		# self.action_space = spaces.Box(low=np.array([0]), high= np.array([+1])) # Beta value 
		# self.observation_space = spaces.Box(low=0, high=1, shape=(np.shape(self.observation)))
		# self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32))

	def createNAgents(self):
		agents = [Agent(self, i) for i in range(self.n)]

		return agents
	
	# Edge E0 - Agent 1 and Agent 2
	# Edge E0 - Agent 1 and Agent 2
	# Edge E0 - Agent 1 and Agent 2

	
	def getState(self,agent_name):
		"""
		Retrieve the state of the network from sumo. 
		"""
		# state = np.zeros(self._num_observation[agent_idx],dtype=np.float32)
		normalizeUniqueVehicleCount = 300
		laneWidthCar = self.traci.lane.getWidth('E0_2')
		laneWidthBike = self.traci.lane.getWidth('E0_1')
		laneWidthPed = self.traci.lane.getWidth('E0_0')
		nLaneWidthCar = np.interp(laneWidthCar, [0,12.6], [0,1])
		nLaneWidthBike = np.interp(laneWidthBike, [0,12.6], [0,1])
		nLaneWidthPed = np.interp(laneWidthPed, [0,12.6], [0,1])

		#E0 is for agent 0 and 1, #-E0 is for agent 2 and 3, #E1 is for agent 4 and 5, #-E1 is for agent 6 and 7
		#E2 is for agent 8 and 9, #-E2 is for agent 10 and 11, #E3 is for agent 12 and 13, #-E3 is for agent 14 and 15

		laneVehicleAllowedType = self.traci.lane.getAllowed('E0_0')
		if 'bicycle' in laneVehicleAllowedType:
			cosharing = 1
		else:
			cosharing = 0

		state = []
		if agent_name == "agent 0": # car
			state_0 = nLaneWidthCar
			state_1 = nLaneWidthBike + nLaneWidthPed
			state_2 = self._total_occupancy_car_Lane*10/self.action_steps			
			state_3 = self._total_density_car_lane*100/self.action_steps
			
			state = [state_0, state_1, state_2, state_3]
			if state_2 > 1 or state_3 > 1:
				print("Agent 0 observation out of bound")

			# state_0 = nLaneWidthCar
			# state_1 = nLaneWidthBike + nLaneWidthPed
			# state_2 = np.interp(self._total_unique_car_count,[0,normalizeUniqueVehicleCount],[0,1]) # average number of unique cars on the car lane through simulation steps
			# state_3 = self._total_occupancy_car_Lane/self.action_steps # average occupancy of the car lane through simulation steps. The raw value is in percentage
			# state_4 = (self._total_occupancy_bike_Lane/self.action_steps + self._total_occupancy_ped_Lane/self.action_steps)/2 # average occupancy of bike plus ped lanes
			# state_5 = np.interp(self._total_count_waiting_car/self.action_steps,[0,10],[0,1]) # average waiting count of cars on the car lane through simulation steps
			# if state_2 == 1 or state_3 == 1 or state_4 == 1 or state_5 == 1:
			# 	print("Agent 0 observation out of bound")


		
		if agent_name == "agent 1": # bike
			state_0 = nLaneWidthBike
			state_1 = nLaneWidthPed				
			state_2 = self._total_occupancy_bike_Lane*100/self.action_steps 		
			state_3 = self._total_occupancy_ped_Lane*100/self.action_steps
		
			state = [state_0, state_1, state_2, state_3]
			if state_2 > 1 or state_3 > 1:
				print("Agent 1 observation out of bound")
		

		if agent_name == "agent 2": 
			state_0 = cosharing #flag for cosharing on or off
			state_7 = np.abs(cosharing-1)
			state_1 = self._total_hinderance_bike_ped/self.action_steps
			state_2 = self._total_hinderance_ped_ped/self.action_steps
			state_3 = self._total_hinderance_bike_bike/self.action_steps
			state_4 = nLaneWidthBike + nLaneWidthPed
			state_5 = self._total_occupancy_bike_Lane*100/self.action_steps 		
			state_6 = self._total_occupancy_ped_Lane*100/self.action_steps
			state_8 = self._total_density_bike_lane*100/self.action_steps 		
			state_9 = self._total_density_ped_lane*100/self.action_steps

			state = [state_0, state_7, state_1, state_2, state_3, state_4, state_5, state_6, state_8, state_9]
			state = [float(self.FlowRateStatsFromRouteFile()[-1])/1200]
			if state_1 == 1 or state_2 == 1:
				print("Agent 2 observation out of bound")

		print(state)
		return np.array(state)
	
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

	def FlowRateStatsFromRouteFile(self):
		tree = ET.parse(self._routeFileName)
		root = tree.getroot()
		vehsPerHour = 0
		bikesPerHour = 0
		pedsPerHour = 0
		for flows in root.iter('flow'):		
			if flows.attrib['id'] == "f_2":
				vehsPerHour = flows.attrib['vehsPerHour']
			elif flows.attrib['id'] == "f_1":
				bikesPerHour = flows.attrib['vehsPerHour']
			elif flows.attrib['id'] == "f_0":
				pedsPerHour = flows.attrib['vehsPerHour']

		return vehsPerHour,bikesPerHour,pedsPerHour

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
		self._total_mean_speed_car = 0
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

	def reset(self,scenario):		
		self._sumo_step = 0
		self._scenario = scenario
		self.resetAllVariables()
		
		if scenario=="Train":
			self._slotId = np.random.randint(1,28)
			#Adapt Route File for continous change
			# self._slotId = 3 # temporary
			# adaptRouteFile(self._slotId, self.pid)
			# if self._slotId < 27:
			# 	self._slotId += 1 
			# else:
			# 	self._slotId = 1
			self._routeFileName = "environment/intersection_Slot_" + str(self._slotId) + ".rou.xml"
			print(self._routeFileName)
		elif scenario=="Test 0":
			self._slotId = np.random.randint(1, 288)
			# self._slotId = 290
			self._routeFileName = "testcase_0/intersection_Slot_" + str(self._slotId) + ".rou.xml"
			print(self._routeFileName)
		else:
			self._slotId = np.random.randint(1, 288)
			self._routeFileName = "testcase_1/intersection_Slot_" + str(self._slotId) + ".rou.xml"
			print(self._routeFileName)
		
		obs_n = []	
		# self.traci.load(['-n', 'environment/intersection.net.xml', '-r', self._routeFileName, "--start"]) # should we keep the previous vehicle
		# if self.firstTimeFlag:
		self.traci.load(self.sumoCMD + ['-n', 'environment/intersection.net.xml', '-r', self._routeFileName])
			# self.firstTimeFlag = False
		# else:
		# 	traci.load(self.sumoCMD + ['-n', 'environment/intersection2.net.xml', '-r', self._routeFileName])
		while self._sumo_step <= self.action_steps:
			self.traci.simulationStep() 		# Take a simulation step to initialize
			self.collectObservation()
			self._sumo_step +=1
		
		self._total_unique_car_count = len(np.unique(np.array(self._unique_car_count_list)))
		self._total_unique_bike_count = len(np.unique(np.array(self._unique_bike_count_list)))
		self._total_unique_ped_count = len(np.unique(np.array(self._unique_ped_count_list)))
		
				
		#record observatinos for each agent
		for agent in self.agents:
			agent.done = False
			obs_n.append(self._get_obs(agent))

		return obs_n

	# get observation for a particular agent
	def _get_obs(self, agent):
		return self.getState(f'agent {agent.id}')

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
		
		# defaultCarLength = 5
		# defaultPedLength = 0.215
		# defaultBikeLength = 1.6
		laneVehicleAllowedType = self.traci.lane.getAllowed('E0_0')
		cosharing = False
		if 'bicycle' in laneVehicleAllowedType: 
			cosharing = True
		if agent.name == "agent 0":
			carLaneWidth = self.traci.lane.getWidth('E0_2')
			if carLaneWidth < 3.2:
				reward = self._fatalPenalty
				agent.done = True
			else:
				#occupancy reward. Lower Occupancy higher reward
				reward_occupancy_car = self._total_density_car_lane/10
				# reward_car_Stopped_count = self._total_count_waiting_car/(self.action_steps*10)
				# print("car stopped: " + str(reward_car_Stopped_count))
				reward = -(reward_occupancy_car)
				# print("agent 0 reward: " + str(reward))
			

		elif agent.name == "agent 1":
			bikeLaneWidth = self.traci.lane.getWidth('E0_1')
			pedLaneWidth = self.traci.lane.getWidth('E0_0')

			if cosharing == True:
				if (bikeLaneWidth + pedLaneWidth) < 2:
					reward = self._fatalPenalty
					agent.done = True
				else:
					reward = self._total_occupancy_ped_Lane*10/(self.action_steps) # as ped lane will count both waiting bikes and peds since the ped lane is coshared and bike lane width = 0
					# print("bike + ped stopped in cosharing: " + str(reward))
					reward = -reward
					# print("agent 1 reward: " + str(reward))
			else:
				if bikeLaneWidth < 1 or pedLaneWidth < 1:
					reward = self._fatalPenalty
					agent.done = True
				else:
					# reward = self._total_count_waiting_ped/(self.action_steps*10) + self._total_count_waiting_bike/(self.action_steps*10)
					reward_occupancy_bike = self._total_occupancy_bike_Lane/self.action_steps
					reward_occupancy_ped = self._total_occupancy_ped_Lane/self.action_steps
					# print("bike + ped stopped: " + str(reward))
					reward = -((reward_occupancy_bike+reward_occupancy_ped)/2)*10
					# print("agent 1 reward: " + str(reward))
		
		elif agent.name == "agent 2":
			# collisionCount = self._collision_count_bike + self._collision_count_ped
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
			densityThreshold = 17
			
			if cosharing:
				if self._total_density_ped_lane > densityThreshold:
					reward = -0.25
				elif self._total_density_ped_lane < densityThreshold:
					reward = +0.5
				
			else:
				if (self._total_density_ped_lane + self._total_density_bike_lane) > 2*densityThreshold:
					reward = +0.5
				elif (self._total_density_ped_lane + self._total_density_bike_lane) < 2*densityThreshold:
					reward = -0.25

			self.reward_agent_2 = reward

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
			# 		# reward = self._fatalPenalty
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
			# # collisionCount = self._collision_count_bike/self.action_steps + self._collision_count_ped/self.action_steps
			# flowrate = float(self.FlowRateStatsFromRouteFile()[-1])/1200
			# if flowrate<0.5 and cosharing:
			# 	reward = 1
			# else:
			# 	reward = -1

			self.reward_agent_2 = reward

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
		self._unique_car_count_list.extend(carsCountList)
		# self._unique_car_count_list.append(carsCountList)
		num_values = len(np.unique(np.array(self._unique_car_count_list)))
		return num_values

	def getUniquePedCount(self):
		pedCountList = self.traci.lane.getLastStepVehicleIDs('E0_0')
		self._unique_ped_count_list.extend(pedCountList)
		num_values = len(np.unique(np.array(self._unique_ped_count_list)))
		return num_values

	def getUniqueBikeCount(self):
		bikeCountList = self.traci.lane.getLastStepVehicleIDs('E0_1')
		self._unique_bike_count_list.extend(bikeCountList)
		num_values = len(np.unique(np.array(self._unique_bike_count_list)))
		return num_values

	def getAllEmergencyBrakingCount(self):		
		allBrakingVehicleIDList = self.traci.simulation.getEmergencyStoppingVehiclesIDList()
		bikeBrakeCounter = 0
		pedBrakeCounter = 0
		carBrakeCounter = 0
		for veh in allBrakingVehicleIDList:
			# print(veh)
			x = veh.split("_",2)
			vehID = x[1].split(".",1)
			if vehID[0]=="1":
				bikeBrakeCounter +=1
			elif vehID[0]=="0":
				pedBrakeCounter +=1
			elif vehID[0]=="2":
				carCollisionCounter +=1

		return carBrakeCounter,bikeBrakeCounter,pedBrakeCounter
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
	
	def collectObservation(self):
		laneWidthCar = self.traci.lane.getWidth('E0_2')
		laneWidthBike = self.traci.lane.getWidth('E0_1')
		laneWidthPed = self.traci.lane.getWidth('E0_0')
		laneVehicleAllowedType = self.traci.lane.getAllowed('E0_0')
		if 'bicycle' in laneVehicleAllowedType:
			cosharing = True
		else:
			cosharing = False
		# record observatinos for each agent
		#Agent 0
		# # Count total number of unique cars on the car lane
		self._unique_car_count_list.extend(self.traci.lane.getLastStepVehicleIDs('E0_2'))
		# Count total occupancy of car lane in percentage
		self._total_occupancy_car_Lane += self.traci.lane.getLastStepOccupancy('E0_2')/laneWidthCar
		# Count total waiting time of the cars in the car lane
		self._total_waiting_time_car += self.traci.lane.getWaitingTime('E0_2')
		# Count total number of cars waiting in the car lane
		self._total_count_waiting_car += self.traci.lane.getLastStepHaltingNumber('E0_2')

		#Returns the mean speed of vehicles that were on this lane within the last simulation step [m/s]
		self._total_mean_speed_car += self.traci.lane.getLastStepMeanSpeed('E0_2')
		self._total_mean_speed_bike += self.traci.lane.getLastStepMeanSpeed('E0_1')
		self._total_mean_speed_ped += self.traci.lane.getLastStepMeanSpeed('E0_0')

		# Count total number of bikes waiting in the bike lane
		self._total_count_waiting_bike += self.traci.lane.getLastStepHaltingNumber('E0_1')
		# Count total number of peds waiting in the ped lane
		self._total_count_waiting_ped += self.traci.lane.getLastStepHaltingNumber('E0_0')


		carCollisionCount, bikeCollisionCount, pedCollisionCount = self.getAllCollisionCount()
		carBrakeCount, bikeBrakeCount, pedBrakeCount = self.getAllEmergencyBrakingCount()
		self._total_density_bike_lane += self.getDensityOfALaneID('E0_1')
		self._total_density_ped_lane += self.getDensityOfALaneID('E0_0')
		self._total_density_car_lane += self.getDensityOfALaneID('E0_2')
		self._EmergencyBraking_count_bike += bikeBrakeCount
		self._EmergencyBraking_count_bike += bikeBrakeCount
		self._EmergencyBraking_count_ped += pedBrakeCount
		
		if cosharing:
			#Agent 1
			# Count total number of unique pedestrian on the ped lane
			# self._total_unique_bike_count += 0 # because this lane width is merged into pedestrian
			# Count total number of unique pedestrian + bike on the ped lane
			# self._total_unique_ped_count += self.getUniquePedCount()
			self._unique_ped_count_list.extend(self.traci.lane.getLastStepVehicleIDs('E0_0'))
			self._unique_bike_count_list.extend(self.traci.lane.getLastStepVehicleIDs('E0_1'))
			self._total_occupancy_bike_Lane += 0
			# Count total occupancy of ped lane in percentage
			self._total_occupancy_ped_Lane += self.traci.lane.getLastStepOccupancy('E0_0')/laneWidthPed

			#Agent 2
			self._collision_count_bike += bikeCollisionCount
			self._collision_count_ped += pedCollisionCount

			# if self._sumo_step % 10 == 0:
			# 	h_b_b, h_b_p, h_p_p =  self.getHinderenaceWhenCosharing('E0_0')
			# 	self._total_hinderance_bike_bike += h_b_b
			# 	self._total_hinderance_bike_ped += h_b_p
			# 	self._total_hinderance_ped_ped += h_p_p
			# print("hinderance bike with ped : " + str(hinderance))

		else:
			#Agent 1
			self._unique_bike_count_list.extend(self.traci.lane.getLastStepVehicleIDs('E0_1'))
			self._unique_ped_count_list.extend(self.traci.lane.getLastStepVehicleIDs('E0_0'))
			# Count total occupancy of bike lane in percentage
			self._total_occupancy_bike_Lane += self.traci.lane.getLastStepOccupancy('E0_1')/laneWidthBike
			# Count total occupancy of ped lane in percentage
			self._total_occupancy_ped_Lane += self.traci.lane.getLastStepOccupancy('E0_0')/laneWidthPed
			# if self._sumo_step % 10 == 0:
			# 	self._total_hinderance_bike_bike += self.getHinderenace('E0_1',"bike_bike")
			# 	self._total_hinderance_ped_ped += self.getHinderenace('E0_0',"ped_ped")

			#Agent 2
			self._collision_count_bike += bikeCollisionCount
			self._collision_count_ped += pedCollisionCount


	def getHinderenaceWhenCosharing(self,laneID):
		h_b_b = 0
		h_b_p = 0
		h_p_p = 0
		bikeList = []
		pedList = []
		
		allVehicles = self.traci.lane.getLastStepVehicleIDs(laneID)			
		if len(allVehicles) > 1:
			for veh in allVehicles:
				x = veh.split("_",2)
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
				x = veh.split("_",2)
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


	def LevelOfService(self,coSharing):
		# It is a function of lane width, total vehicle number, hindrance_bb,hinderence_cc,hindrance_bc}
		self.w_lane_width = 0.3
		self.w_total_occupancy = 0.3
		
		if coSharing:
			self.w_hinderance_b_b = 0.1
			self.w_hinderance_b_p = 0.2
			self.w_hinderance_p_p = 0.1
			laneID = 'E0_0'
			laneWidth = self.traci.lane.getWidth(laneID)
			los = -self.w_lane_width*laneWidth + self.w_total_occupancy*self._total_occupancy_ped_Lane  + self.w_hinderance_b_b*self._total_hinderance_bike_bike + \
				 self.w_hinderance_b_p*self._total_hinderance_bike_ped + self.w_hinderance_p_p*self._total_hinderance_ped_ped

		else:
			self.w_hinderance_b_b = 0.2
			self.w_hinderance_b_p = 0
			self.w_hinderance_p_p = 0.2
			pedLaneID = 'E0_0'
			bikeLaneID = 'E0_1'
			pedLaneWidth = self.traci.lane.getWidth(pedLaneID)
			bikeLaneWidth = self.traci.lane.getWidth(bikeLaneID)
			los_ped_Lane = -self.w_lane_width*pedLaneWidth + self.w_total_occupancy*self._total_occupancy_ped_Lane  + self.w_hinderance_p_p*self._total_hinderance_ped_ped
			los_bike_Lane = -self.w_lane_width*bikeLaneWidth + self.w_total_occupancy*self._total_occupancy_bike_Lane  + self.w_hinderance_b_b*self._total_hinderance_bike_bike
			los = (los_ped_Lane + los_bike_Lane)/2


		return los

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
		# adaptRouteFile(self._slotId, self.pid)
		# self.traci.load(self.sumoCMD + ['-n', 'environment/intersection.net.xml', '-r', self._routeFileName])
		#simulating a warm period of N=self.action_steps  and then recording the state, action, reward tuple. 
		bikeLaneWidth = self.traci.lane.getWidth('E0_1')
		pedlLaneWidth = self.traci.lane.getWidth('E0_0')
		carLaneWidth = self.traci.lane.getWidth('E0_2')
		
		
		laneVehicleAllowedType = self.traci.lane.getAllowed('E0_0')
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
				if agent.name == "agent 2":
					self.coShareValue = action_n[i]
			self._set_action(temp_action_dict,agent,action_space_dict)
			actionFlag = False
		if self._scenario == "Train" or self._scenario == "Test 0":
			#reset all variables
			self.resetAllVariables()
			if 'bicycle' in laneVehicleAllowedType:
				cosharing = True
			else:
				cosharing = False
			while self._sumo_step <= self.action_steps:
				# advance world state	
				self.traci.simulationStep()
				self._sumo_step +=1
				self.collectObservation()
			
			self._total_unique_car_count = len(np.unique(np.array(self._unique_car_count_list).flatten()))
			self._total_unique_bike_count = len(np.unique(np.array(self._unique_bike_count_list).flatten()))
			self._total_unique_ped_count = len(np.unique(np.array(self._unique_ped_count_list).flatten()))

			print("Hinderance bike with bike : " + str(self._total_hinderance_bike_bike))
			print("Hinderance bike with ped : " + str(self._total_hinderance_bike_ped))
			print("Hinderance ped with ped : " + str(self._total_hinderance_ped_ped))
			self._levelOfService = self.LevelOfService(cosharing)
			print("Level of Service : " + str(self.LevelOfService(cosharing)))
			
		# 	if 'bicycle' in laneVehicleAllowedType:
		# 		cosharing = True
		# 	else:
		# 		cosharing = False
		# 	# record observatinos for each agent
		# 	#Agent 0
		# 	# # Count total number of unique cars on the car lane
		# 	self._unique_car_count_list.append(self.traci.lane.getLastStepVehicleIDs('E0_2'))
		# 	# Count total occupancy of car lane in percentage
		# 	self._total_occupancy_car_Lane += self.traci.lane.getLastStepOccupancy('E0_2')
		# 	# Count total waiting time of the cars in the car lane
		# 	self._total_waiting_time_car += self.traci.lane.getWaitingTime('E0_2')
		# 	# Count total number of cars waiting in the car lane
		# 	self._total_count_waiting_car += self.traci.lane.getLastStepHaltingNumber('E0_2')
		# 	# Count total number of bikes waiting in the bike lane
		# 	self._total_count_waiting_bike += self.traci.lane.getLastStepHaltingNumber('E0_1')
		# 	# Count total number of peds waiting in the ped lane
		# 	self._total_count_waiting_ped += self.traci.lane.getLastStepHaltingNumber('E0_0')


		# 	carCollisionCount, bikeCollisionCount, pedCollisionCount = self.getAllCollisionCount()
			
			
		# 	if cosharing:
		# 		#Agent 1
		# 		# Count total number of unique pedestrian on the ped lane
		# 		# self._total_unique_bike_count += 0 # because this lane width is merged into pedestrian
		# 		# Count total number of unique pedestrian + bike on the ped lane
		# 		# self._total_unique_ped_count += self.getUniquePedCount()
		# 		self._unique_ped_count_list.append(self.traci.lane.getLastStepVehicleIDs('E0_0'))
		# 		self._unique_bike_count_list.append(self.traci.lane.getLastStepVehicleIDs('E0_1'))
		# 		self._total_occupancy_bike_Lane += 0
		# 		# Count total occupancy of ped lane in percentage
		# 		self._total_occupancy_ped_Lane += self.traci.lane.getLastStepOccupancy('E0_0')

		# 		#Agent 2
		# 		self._collision_count_bike += bikeCollisionCount
		# 		self._collision_count_ped += pedCollisionCount
				

		# 	else:
		# 		#Agent 1
		# 		self._unique_bike_count_list.append(self.traci.lane.getLastStepVehicleIDs('E0_1'))
		# 		self._unique_ped_count_list.append(self.traci.lane.getLastStepVehicleIDs('E0_0'))
		# 		# Count total occupancy of bike lane in percentage
		# 		self._total_occupancy_bike_Lane += self.traci.lane.getLastStepOccupancy('E0_1')
		# 		# Count total occupancy of ped lane in percentage
		# 		self._total_occupancy_ped_Lane += self.traci.lane.getLastStepOccupancy('E0_0')

		# 		#Agent 2
		# 		self._collision_count_bike += bikeCollisionCount
		# 		self._collision_count_ped += pedCollisionCount

		# 	# # # if rewardFlag == True and (self._sumo_step-2)%90 == 0:
		# 	# self._averageRewardStepCounter = self._averageRewardStepCounter + 1
		# 	# queue_length, queue_Count = self.getQueueLength('E0_2')
		# 	# self._queue_Length_car_agent_0 += queue_length
		# 	# self._queue_Count_car_agent_0 += queue_Count
		# 	# queue_length,queue_Count = self.getQueueLength('E0_0')
		# 	# self._queue_Length_ped_agent_1 += queue_length
		# 	# self._queue_Count_ped_agent_1 += queue_Count
		# 	# queue_length,queue_Count = self.getQueueLength('E0_1')
		# 	# self._queue_Length_bike_agent_1 += queue_length
		# 	# self._queue_Count_bike_agent_1 += queue_Count

			
			
		# 	# bikes = self.traci.lane.getLastStepVehicleNumber('E0_1') # Number of vehicles on that lane 
		# 	# self._total_bike_on_lane_agent_1 += bikes
		# 	# peds = self.traci.lane.getLastStepVehicleNumber('E0_0') # Number of vehicles on that lane 
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
			obs_n.append(self._get_obs(agent))	
			# print(self._get_obs(agent))		
			reward_n.append(self._get_reward(agent))
			# print(self._get_reward(agent))
			done_n.append(self._get_done(agent))

			info_n['n'].append(self._get_info(agent))

		# # all agents get total reward in cooperative case
		# # if self._fatalErroFlag == True:
		# # 	reward = 0
		# self._carQueueLength += self._queue_Length_car_agent_0 / self._averageRewardStepCounter
		# self._bikeQueueLength += self._queue_Length_bike_agent_1 / self._averageRewardStepCounter
		# self._pedQueueLength += self._queue_Length_ped_agent_1 / self._averageRewardStepCounter
		# else:
		self._currentReward = reward_n
		reward = np.sum(reward_n)
		if self.shared_reward:
			reward_n = [reward] *self.n
		print("Reward = " + str(reward_n))
		self._lastReward = reward_n[0]
		print("reward: " + str(self._lastReward))
		# print("Number of cars passed: " + str(self._total_vehicle_passed))
		return obs_n, reward_n, done_n, info_n


	def rewardAnalysisStats(self):			
		# return self._currentReward[0],self._currentReward[1]
		return self._currentReward[0],self._currentReward[1],self._currentReward[2]

	# set env action for a particular agent
	def _set_action(self, actionDict, agent, action_space, time=None):
		# process action
		t = 0
		adaptNetwork(self.base_netfile,actionDict,agent.name,self._routeFileName,self.sumoCMD, self.pid, self.traci)
		# t = 0
	def testAnalysisStats(self):
		bikeLaneWidth = self.traci.lane.getWidth('E0_1')
		pedlLaneWidth = self.traci.lane.getWidth('E0_0')
		carLaneWidth = self.traci.lane.getWidth('E0_2')
		laneVehicleAllowedType = self.traci.lane.getAllowed('E0_0')
		cosharing = 999
		if 'bicycle' in laneVehicleAllowedType:
			cosharing = 1
		else:
			cosharing = 0

		
		self._carFlow,self._bikeFlow,self._pedFlow = self.FlowRateStatsFromRouteFile()
		
		return self._carFlow,self._bikeFlow,self._pedFlow,carLaneWidth,bikeLaneWidth,pedlLaneWidth,cosharing,self._total_mean_speed_car,\
			self._total_mean_speed_bike,self._total_mean_speed_ped,self._total_count_waiting_car,self._total_count_waiting_bike,\
				self._total_count_waiting_ped,self._total_unique_car_count,self._total_unique_bike_count,self._total_unique_ped_count,\
					self._total_occupancy_car_Lane,self._total_occupancy_bike_Lane,self._total_occupancy_ped_Lane,self._collision_count_bike,\
						self._collision_count_ped,self._total_density_bike_lane,self._total_density_ped_lane,self._total_density_car_lane, \
						self._total_hinderance_bike_bike,self._total_hinderance_bike_ped,self._total_hinderance_ped_ped,self._levelOfService


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
		self.sumoCMD = ["--time-to-teleport.disconnected",str(1),"--ignore-route-errors","--device.rerouting.probability","1","--device.rerouting.period","1",
						"--pedestrian.striping.dawdling","0.5","--collision.check-junctions", str(True),"--collision.mingap-factor","0","--collision.action", "warn",
						 "--seed", f"{np.random.randint(69142)}", "-W","--default.carfollowmodel", "IDM","--no-step-log","--statistic-output","output.xml"]
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