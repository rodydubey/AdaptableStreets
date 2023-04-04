# adaptNetwork.py

import xml.etree.ElementTree as ET
from sumolib import checkBinary
import sys
import numpy as np
import subprocess

baselineCarLaneWidth = 9.6
baselinebicycleLaneWidth = 1.5
baselinePedestrianLaneWidth = 1.5
totalEdgeWidth = baselineCarLaneWidth + baselinebicycleLaneWidth + baselinePedestrianLaneWidth
carLane_width_actions = ['3.2','5.4','6.6','7.8','9.6']
bikeLane_width_actions = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']

netconvert = checkBinary("netconvert")
sys.path.append(netconvert)

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

#function
def adaptNetwork(sumo_edges, base_network,actionDict,modelType,routeFileName,sumoCMD, pid, traci):
    # parsing directly.
    tree = ET.parse(base_network)
    root = tree.getroot()
    
    remainderLaneLength = 0
    edge_props = {}

    for (key, edge_id), value in actionDict.items():
        props = edge_props.get(edge_id,{})
        if key == "agent 0":
            
            if modelType == "Heuristic":
                carLaneWidth = value
            else:
                carLaneWidth = float(carLane_width_actions[value])
                remainderLaneLength = totalEdgeWidth - carLaneWidth
            props['carLaneWidth'] = carLaneWidth
        elif key == "agent 1":
            if modelType == "Heuristic":
                bikeLaneWidth = value
                pedLaneWidth = float(totalEdgeWidth-(carLaneWidth + bikeLaneWidth))
            else:            
                bikeLaneWidth = float(bikeLane_width_actions[value])*remainderLaneLength
                pedLaneWidth = float(totalEdgeWidth-(carLaneWidth + bikeLaneWidth))
            props['bikeLaneWidth'] = bikeLaneWidth
            props['pedLaneWidth'] = pedLaneWidth
        elif key == "agent 2":   
            if value < 1:
                coShare = 0    
            else:
                coShare = 1  
            props['coShare'] = coShare
        edge_props[edge_id] = props    
    # carLaneWidth_agent_0 = 6.2
    # bikeLaneWidth_agent_1 = 3.2
    # pedLaneWidth_agent_1 = 3.2
    # coShare = 0.1
    #E0 is for agent 0 and 1, #-E0 is for agent 2 and 3, #E1 is for agent 4 and 5, #-E1 is for agent 6 and 7
    #E2 is for agent 8 and 9, #-E2 is for agent 10 and 11, #E3 is for agent 12 and 13, #-E3 is for agent 14 and 15
    # coShare = 0.6
    # if coShare > 0.5:
    #     coShare = 0.1
    # for agent_edge in sumo_edges:
    print(edge_props)
    for edge_id, props in edge_props.items():
        carLaneWidth = props['carLaneWidth']
        bikeLaneWidth = props['bikeLaneWidth']
        pedLaneWidth = props['pedLaneWidth']
        coShare = props['coShare']
        if coShare <= 0.5:
            for lanes in root.iter('lane'):
                if lanes.attrib['id'] == f"{edge_id}_2":
                    lanes.attrib['width'] = repr(carLaneWidth)
                    # lanes.attrib['width'] = carLaneWidth
            
                elif lanes.attrib['id'] == f"{edge_id}_1":
                    lanes.attrib['width'] = repr(bikeLaneWidth)
                    # lanes.attrib['width'] = bikeWidthTemp

                elif lanes.attrib['id'] == f"{edge_id}_0":
                    lanes.attrib['width'] = repr(pedLaneWidth)
                    # lanes.attrib['width'] = bikeWidthTemp
        else:
            for lanes in root.iter('lane'):
                if lanes.attrib['id'] == f"{edge_id}_2":
                    lanes.attrib['width'] = repr(carLaneWidth)
                    # lanes.attrib['width'] = carLaneWidth
            
                elif lanes.attrib['id'] == f"{edge_id}_0":
                    lanes.attrib['width'] = repr(bikeLaneWidth+pedLaneWidth)
                    # lanes.attrib['width'] = bikeWidthTemp

                elif lanes.attrib['id'] == f"{edge_id}_1":
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
    subprocess.run(f"netconvert -s {modified_netfile} -o {modified_netfile} -W", capture_output=True, shell=True)
    # allVehicles = traci.vehicle.getIDList()
   
    
        
    # save state
    # traci.simulation.saveState('environment/savedstate.xml') 
    # load traci simulation   
    # traci.load(['-n', "environment\intersection2.net.xml","--start"])

    # traci.load(sumoCMD + ['-n', 'environment/intersection2.net.xml', '-r', routeFileName, '--additional-files',"environment/intersection2.add.xml"])
    traci.load(sumoCMD + ['-n', modified_netfile, '-r', routeFileName])
    # traci.load(['-n', 'environment/intersection2.net.xml', '-r', routeFileName, "--start"]) # should we keep the previous vehic
   
    # load last saved state
    # traci.simulation.loadState('environment/savedstate.xml')

    for edge_id, props in edge_props.items():
        carLaneWidth = props['carLaneWidth']
        bikeLaneWidth = props['bikeLaneWidth']
        pedLaneWidth = props['pedLaneWidth']
        coShare = props['coShare']
        #change lane sharing based on agent choice
        if coShare <= 0.5:
            # print(str(coShare) + "--- NO Co-Sharing")
            disallowed = ['private', 'emergency', 'passenger','authority', 'army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
            disallowed.append('pedestrian')
            traci.lane.setDisallowed(f'{edge_id}_1',disallowed)
            traci.lane.setAllowed(f'{edge_id}_1','bicycle')
            disallowed2 = ['private', 'emergency', 'passenger', 'authority', 'army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
            disallowed2.append('bicycle')
            traci.lane.setDisallowed(f'{edge_id}_0',disallowed2)
            traci.lane.setAllowed(f'{edge_id}_0','pedestrian')
        else: 
            # print(str(coShare) + "--- YES Co-Sharing")
            disallowed3 = ['private', 'emergency', 'authority', 'passenger','army', 'vip', 'hov', 'taxi', 'bus', 'coach', 'delivery', 'truck', 'trailer', 'motorcycle', 'moped', 'evehicle', 'tram', 'rail_urban', 'rail', 'rail_electric', 'rail_fast', 'ship', 'custom1', 'custom2']
            disallowed3.append('bicycle')
            disallowed3.append('pedestrian')
            traci.lane.setDisallowed(f'{edge_id}_0',disallowed3)
            allowed = []
            allowed.append('bicycle')
            allowed.append('pedestrian')        
            traci.lane.setAllowed(f'{edge_id}_0',allowed)
            # peds= traci.lane.getLastStepVehicleIDs(f"{edge_id}_0")
            # allVehicles = traci.vehicle.getIDList()
            traci.lane.setDisallowed(f'{edge_id}_1', ["all"])
            # traci.lane.setAllowed(f'{agent_edge}_0','bicycle')
            #loop through all pedestrian on E0_0 lane and change lane to E0_1
            # car_list = traci.vehicle.getIDList()
