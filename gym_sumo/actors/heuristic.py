from gym_sumo.actors.agent import Agent

import numpy as np

baselineCarLaneWidth = 9.6
baselinebicycleLaneWidth = 1.5
baselinePedestrianLaneWidth = 1.5
totalEdgeWidth = baselineCarLaneWidth + baselinebicycleLaneWidth + baselinePedestrianLaneWidth

class HeuristicAgent(Agent):
    def act(self, states, epnum):
        states = [{'state': sk} for sk in states]
        result = []

        car_length = 5
        ped_length = 0.215
        bike_length = 1.6
        carflow = max(0.01, self.env._total_occupancy_car_Lane/car_length)
        pedflow = max(0.01, self.env._total_occupancy_ped_Lane/ped_length)
        bikeflow = max(0.01, self.env._total_occupancy_bike_Lane/bike_length)
        all_flows = [carflow, pedflow, bikeflow]
        print('all flows',all_flows)
        alpha = np.clip(carflow/sum(all_flows), 0.1,0.9)
        carLaneWidth = min(max(3.2, alpha*totalEdgeWidth), 10.6)
        alpha = carLaneWidth/totalEdgeWidth

        remainderRoad_0 = totalEdgeWidth - carLaneWidth
        beta = np.clip(bikeflow/sum(all_flows[1:]), 0.1,0.9)
        bikeLaneWidth = max(1.5, beta*remainderRoad_0)
        beta = bikeLaneWidth/remainderRoad_0

        densityThreshold = 1
        if (self.env._total_density_ped_lane + self.env._total_density_bike_lane) > 2*densityThreshold:
            coshare = 0
        else:
            coshare = 1

        print('COSHARE', coshare)
        actions = [alpha, beta, coshare]
        for i, action in enumerate(actions):
            if i==2:
                action_prob = np.array([[action==0, action==1]])*1
            else:
                action_prob = np.array([[action]])
            result.append((np.array([[action]]), action_prob))
        actions = [r[0][0] for r in result]
        action_probs = [r[1] for r in result]
        return actions, action_probs

    def train(self):
        pass

    def store_transition(self, old_states, action_probs, states, rewards, terminals):
        pass

    def process_transitions(self):
        pass

    def get_buffer_size(self):
        return -1