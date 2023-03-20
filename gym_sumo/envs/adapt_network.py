# adaptNetwork.py

import xml.etree.ElementTree as ET
try:
	import libsumo as traci
except:
	import traci
from sumolib import checkBinary
import os
import sys
import numpy as np
baselineCarLaneWidth = 9.6
baselinebicycleLaneWidth = 1.5
baselinePedestrianLaneWidth = 1.5
totalEdgeWidth = baselineCarLaneWidth + baselinebicycleLaneWidth + baselinePedestrianLaneWidth

netconvert = checkBinary("netconvert")
sys.path.append(netconvert)

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

#function
def adaptNetwork(base_network,actionDict,name,routeFileName,sumoCMD, pid):
    # parsing directly.
    tree = ET.parse(base_network)
    root = tree.getroot()
   
    remainderRoad_0 = 0
    carLaneWidth_agent_0 = 9.6
    bikeLaneWidth_agent_1 = 1.5
    pedLaneWidth_agent_1 = 1.5

   

    for key, value in actionDict.items():
        if key == "agent 0":
            alpha = value  
            # alpha = random.uniform(0.1, 0.9)
            # alpha = 0.5
            carLaneWidth_agent_0 = float(alpha*totalEdgeWidth)
            # carLaneWidth_agent_0 = clamp(carLaneWidth_agent_0,3.2,9.6)
            remainderRoad_0 = totalEdgeWidth - carLaneWidth_agent_0
        elif key == "agent 1":
            beta = value
            # beta = clamp(beta,0.1,0.99)
            # beta = random.uniform(0.2, 0.8)
            # beta = 0.5
            bikeLaneWidth_agent_1 = float(beta*remainderRoad_0)
            if bikeLaneWidth_agent_1 == 0:
                breakk = 0
            pedLaneWidth_agent_1 = float((1-beta)*remainderRoad_0)

        elif key == "agent 2":
            coShare = np.round(float(value))     
            # coShare = random.uniform(0.1, 0.9)            
            
        
    # carLaneWidth_agent_0 = 6.2
    # bikeLaneWidth_agent_1 = 3.2
    # pedLaneWidth_agent_1 = 3.2
    # coShare = 0.1
    #E0 is for agent 0 and 1, #-E0 is for agent 2 and 3, #E1 is for agent 4 and 5, #-E1 is for agent 6 and 7
    #E2 is for agent 8 and 9, #-E2 is for agent 10 and 11, #E3 is for agent 12 and 13, #-E3 is for agent 14 and 15
    # coShare = 0.6
    # if coShare > 0.5:
    #     coShare = 0.1
    if coShare <= 0.5:
        for lanes in root.iter('lane'):
            if lanes.attrib['id'] == "E0_2":
                lanes.attrib['width'] = repr(carLaneWidth_agent_0)
                # lanes.attrib['width'] = carLaneWidth
           
            elif lanes.attrib['id'] == "E0_1":
                lanes.attrib['width'] = repr(bikeLaneWidth_agent_1)
                # lanes.attrib['width'] = bikeWidthTemp

            elif lanes.attrib['id'] == "E0_0":
                lanes.attrib['width'] = repr(pedLaneWidth_agent_1)
                # lanes.attrib['width'] = bikeWidthTemp
    else:
        for lanes in root.iter('lane'):
            if lanes.attrib['id'] == "E0_2":
                lanes.attrib['width'] = repr(carLaneWidth_agent_0)
                # lanes.attrib['width'] = carLaneWidth
           
            elif lanes.attrib['id'] == "E0_0":
                lanes.attrib['width'] = repr(bikeLaneWidth_agent_1+pedLaneWidth_agent_1)
                # lanes.attrib['width'] = bikeWidthTemp

            elif lanes.attrib['id'] == "E0_1":
                lanes.attrib['width'] = repr(0)
                # lanes.attrib['width'] = bikeWidthTemp
        
        
    #  write xml 
    modified_netfile = f'environment/intersection2_{pid}.net.xml'
    file_handle = open(modified_netfile,"wb")
    tree.write(file_handle)
    file_handle.close()
    # call netconvert            
    # os.system("C:/D/SUMO/SumoFromSource/bin/netconvert.exe -s environment\intersection2.net.xml -o environment\intersection2.net.xml --crossings.guess")
    # os.system("C:/D/SUMO/SumoFromSource/bin/netconvert.exe -s environment\intersection2.net.xml -o environment\intersection2.net.xml")
    # netconvert = checkBinary("netconvert")
    os.system(f"netconvert -s {modified_netfile} -o {modified_netfile} -W")
    # allVehicles = traci.vehicle.getIDList()
    
    # peds= traci.lane.getLastStepVehicleIDs("E0_0")
    # for ped_id in allVehicles:
    #     traci.vehicle.changeLane(ped_id,1, 3000)

    # if coShare <= 0.5:
    #     # print(str(coShare) + "--- NO Co-Sharing")
    #     disallowed = ['private', 'emergency', 'passenger','authority', 'army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
    #     disallowed.append('pedestrian')
    #     traci.lane.setDisallowed('E0_1',disallowed)
    #     traci.lane.setAllowed('E0_1','bicycle')
    #     disallowed2 = ['private', 'emergency', 'passenger', 'authority', 'army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
    #     disallowed2.append('bicycle')
    #     traci.lane.setDisallowed('E0_0',disallowed2)
    #     traci.lane.setAllowed('E0_0','pedestrian')
    # else: 
    #     # print(str(coShare) + "--- YES Co-Sharing")
    #     disallowed3 = ['private', 'emergency', 'authority', 'passenger','army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
    #     disallowed3.append('bicycle')
    #     disallowed3.append('pedestrian')
    #     traci.lane.setDisallowed('E0_1',disallowed3)
    #     allowed = []
    #     allowed.append('bicycle')
    #     allowed.append('pedestrian')        
    #     traci.lane.setAllowed('E0_1',allowed)        
    #     # traci.lane.setDisallowed('E0_0',disallowed3)

    # peds= traci.lane.getLastStepVehicleIDs("E0_0")
    # for ped_id in allVehicles:
    #     traci.vehicle.changeLane(ped_id,1, 3000)
   
    
        
    # save state
    # traci.simulation.saveState('environment/savedstate.xml') 
    # load traci simulation   
    # traci.load(['-n', "environment\intersection2.net.xml","--start"])

    # traci.load(sumoCMD + ['-n', 'environment/intersection2.net.xml', '-r', routeFileName, '--additional-files',"environment/intersection2.add.xml"])
    traci.load(sumoCMD + ['-n', modified_netfile, '-r', routeFileName])
    # traci.load(['-n', 'environment/intersection2.net.xml', '-r', routeFileName, "--start"]) # should we keep the previous vehic
   
    # load last saved state
    # traci.simulation.loadState('environment/savedstate.xml')

    #change lane sharing based on agent choice
    if coShare <= 0.5:
        # print(str(coShare) + "--- NO Co-Sharing")
        disallowed = ['private', 'emergency', 'passenger','authority', 'army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
        disallowed.append('pedestrian')
        traci.lane.setDisallowed('E0_1',disallowed)
        traci.lane.setAllowed('E0_1','bicycle')
        disallowed2 = ['private', 'emergency', 'passenger', 'authority', 'army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
        disallowed2.append('bicycle')
        traci.lane.setDisallowed('E0_0',disallowed2)
        traci.lane.setAllowed('E0_0','pedestrian')
    else: 
        # print(str(coShare) + "--- YES Co-Sharing")
        disallowed3 = ['private', 'emergency', 'authority', 'passenger','army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
        disallowed3.append('bicycle')
        disallowed3.append('pedestrian')
        traci.lane.setDisallowed('E0_0',disallowed3)
        allowed = []
        allowed.append('bicycle')
        allowed.append('pedestrian')        
        traci.lane.setAllowed('E0_0',allowed)
        # peds= traci.lane.getLastStepVehicleIDs("E0_0")
        # allVehicles = traci.vehicle.getIDList()
        traci.lane.setDisallowed('E0_1', ["all"])
        # traci.lane.setAllowed('E0_0','bicycle')
         #loop through all pedestrian on E0_0 lane and change lane to E0_1
        # car_list = traci.vehicle.getIDList()
        
        # laneIndex = 1
        # # for ped_id in peds:
        # #     traci.vehicle.remove(ped_id)
        # pendingVehicles = traci.simulation.getPendingVehicles()
        # bikes = traci.lane.getLastStepVehicleIDs("E0_0")
        # for bike_id in bikes:
        #     traci.vehicle.changeLane(bike_id,0,4000)

        # peds = traci.lane.getLastStepVehicleIDs("E0_1")
     
        # peds = traci.lane.getLastStepVehicleIDs("E0_0")
        
        # traci.lane.setDisallowed('E0_0',disallowed3)
        # if "f_0.17" in traci.vehicle.getIDList():
        #     traci.vehicle.remove("f_0.17")

    # bikes = traci.vehicle.getIDList()
    # for bike_id in bikes:
    #     routeID = traci.vehicle.getRoute(bike_id)
    #     print(routeID)

    # print('E0_0: ' + str(traci.lane.getAllowed('E0_0')))
    # print('E0_1: ' + str(traci.lane.getAllowed('E0_1')))