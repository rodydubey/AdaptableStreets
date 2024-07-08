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
# carLane_width_actions = ['3.2','5.4','6.6','7.8','9.6']
carLane_width_actions = ['3.2','5.6','6.4','7.8','9.6']
bikeLane_width_actions = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
netconvert = checkBinary("netconvert")

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def vehFallOutsideItsLane(pos,shape,width,veh):
    y2 = shape[0][1]+width/2
    y1 = shape[0][1]-width/2

    x, y = pos[0],pos[1]
    if (y1 < y and y < y2):
        return False
    return True

def computeSafetyMeasure(traci,env,coShare_ForSafetyCheck,pedLaneWidth,bikeLaneWidth,carLaneWidth):
    pedCounter = 0
    bikeCounter = 0
    vehCounter = 0
    edge_id = 'E0'
    allVehicles = traci.edge.getLastStepVehicleIDs(edge_id)
    for veh in allVehicles:
        pos = traci.vehicle.getPosition(veh)
        vehType = traci.vehicle.getTypeID(veh)
        if vehType=="ped":
            #position of pedLane
            pedLaneForThisEdge = f"{edge_id}_0"
            # width = traci.lane.getWidth(pedLaneForThisEdge)
            shape = traci.lane.getShape(pedLaneForThisEdge)
            _pedwidth_shape = pedLaneWidth
            if coShare_ForSafetyCheck > 0.5:
                _pedwidth_shape = pedLaneWidth + bikeLaneWidth
            if vehFallOutsideItsLane(pos,shape,_pedwidth_shape,veh): #if True than it is potential issue
                if pos[1]<(shape[0][1]-_pedwidth_shape/2): # out on grass for E0
                    continue
                pedCounter+=1
                # print(veh,"-")
        elif vehType=="bike": 
                _bikewidth_shape = 0  
                if coShare_ForSafetyCheck <= 0.5:
                    bikeLaneForThisEdge = f"{edge_id}_1"
                    shape = traci.lane.getShape(bikeLaneForThisEdge)
                    _bikewidth_shape = bikeLaneWidth
                else:
                    bikeLaneForThisEdge = f"{edge_id}_0"
                    shape = traci.lane.getShape(bikeLaneForThisEdge)
                # width = traci.lane.getWidth(pedLaneForThisEdge)
                    _bikewidth_shape = bikeLaneWidth + pedLaneWidth
            
                if vehFallOutsideItsLane(pos,shape,_bikewidth_shape,veh): #if True than it is potential issue
                    bikeCounter+=1
                    # print(veh,"-")
        else:
            vehLaneForThisEdge = f"{edge_id}_2"
            # width = traci.lane.getWidth(pedLaneForThisEdge)
            shape = traci.lane.getShape(vehLaneForThisEdge)
            if vehFallOutsideItsLane(pos,shape,carLaneWidth,veh): #if True than it is potential issue
                vehCounter+=1
                # print(veh,"-")
    ###############################
    env.pedSafetyCounter = pedCounter
    env.bikeSafetyCounter = bikeCounter
    env.vehSafetyCounter = vehCounter
    print(pedCounter,"-",bikeCounter,"-",vehCounter)

#function
def adaptNetwork(env, sumo_edges, base_network,actionDict,modelType,routeFileName,sumoCMD, pid, traci):
    if modelType != 'static':
        # parsing directly.
        tree = ET.parse(base_network)
        root = tree.getroot()
        
        remainderLaneLength = 0
        edge_props = {}

        for (key, edge_id), value in actionDict.items():
            props = edge_props.get(edge_id,{})
            if "agent 0" in key:
                
                if modelType == "heuristic":
                    carLaneWidth = value
                
                else:
                    carLaneWidth = float(carLane_width_actions[value])
                    remainderLaneLength = totalEdgeWidth - carLaneWidth
                props['carLaneWidth'] = carLaneWidth
            elif "agent 1" in key:
                if modelType == "heuristic":
                    bikeLaneWidth = value
                    pedLaneWidth = float(totalEdgeWidth-(carLaneWidth + bikeLaneWidth))
                else:            
                    bikeLaneWidth = float(bikeLane_width_actions[value])*remainderLaneLength
                    pedLaneWidth = float(totalEdgeWidth-(carLaneWidth + bikeLaneWidth))
                props['bikeLaneWidth'] = bikeLaneWidth
                props['pedLaneWidth'] = pedLaneWidth
            elif "agent 2" in key:   
                if value < 1:
                    coShare = 0    
                else:
                    coShare = 1  
                props['coShare'] = coShare
            edge_props[edge_id] = props
            if len(sumo_edges)!=5: # NOTE: CHECK IF CAUSES PROBLEMS, HACK FOR BARCELONA
                from_tos = {'E0': 'E2',
                            '-E1': 'E3',
                            '-E2': '-E0',
                            '-E3': 'E1'}
            else:
                from_tos = {'803424574#0': '237645185#0',
                            '237645196#0': '525638416#0',
                            '237790228#0': '237790228#3',
                            '237645189#0': '803424599#1',
                            '237910181#3': '544248640#2'}       
            edge_props[from_tos[edge_id]] = props # also set properties of downstream

        for edge_id, props in edge_props.items():
            carLaneWidth = props['carLaneWidth']
            bikeLaneWidth = props['bikeLaneWidth']
            pedLaneWidth = props['pedLaneWidth']
            coShare = props['coShare']
            coShare_ForSafetyCheck = coShare
            for lanes in root.iter('lane'):
                if lanes.attrib['id'] == f"{edge_id}_2":
                    lanes.attrib['width'] = repr(carLaneWidth)
                if coShare <= 0.5:            
                    if lanes.attrib['id'] == f"{edge_id}_1":
                        lanes.attrib['width'] = repr(bikeLaneWidth)
                        lanes.attrib.pop('disallow', None)
                        lanes.attrib['allow'] = 'bicycle'

                    elif lanes.attrib['id'] == f"{edge_id}_0":
                        lanes.attrib['width'] = repr(pedLaneWidth)
                        lanes.attrib.pop('disallow', None)
                        lanes.attrib['allow'] = 'pedestrian'
                else:             
                    if lanes.attrib['id'] == f"{edge_id}_0":
                        lanes.attrib['width'] = repr(bikeLaneWidth+pedLaneWidth)
                        lanes.attrib.pop('disallow', None)
                        lanes.attrib['allow'] = 'bicycle pedestrian'
                    elif lanes.attrib['id'] == f"{edge_id}_1":
                        lanes.attrib['width'] = repr(0)
                        lanes.attrib['disallow'] = 'all'
                        lanes.attrib.pop('allow', None)

        #  write xml 
        modified_netfile = f'environment/intersection2_{pid}.net.xml'
        file_handle = open(modified_netfile,"wb")
        tree.write(file_handle)
        file_handle.close()
        if modified_netfile not in env.generatedFiles:
            env.generatedFiles.append(modified_netfile)
        subprocess.run(f"{netconvert} -s {modified_netfile} -o {modified_netfile} -W", capture_output=True, shell=True)
    else:
        modified_netfile = base_network
        env.pedSafetyCounter = 0
        env.bikeSafetyCounter = 0
        env.vehSafetyCounter = 0

    # call netconvert            
    # netconvert = checkBinary("netconvert")

    # save state
    if env.load_state:
        env.state_file = f'environment/savedstate_{pid}.xml'
        traci.simulation.clearPending()
        traci.simulation.saveState(env.state_file) 
    # load traci simulation   
    # traci.load(['-n', "environment\intersection2.net.xml","--start"])
    currentTime = (env.timeOfHour-1)*6*300 # TODO: fix hardcoded

    # traci.load(sumoCMD + ['-n', 'environment/intersection2.net.xml', '-r', routeFileName, '--additional-files',"environment/intersection2.add.xml"])
    additional_args = ['-n', modified_netfile, '-r', routeFileName]

    # if env._scenario == "Test Single Flow":
    #     additional_args += ['-b', str(currentTime)]
    traci.load(sumoCMD + additional_args)
   

    # traci.load(['-n', 'environment/intersection2.net.xml', '-r', routeFileName, "--start"]) # should we keep the previous vehic
    # if env._scenario == "Test Single Flow":
    #     traci.simulationStep(currentTime)
    # load last saved state
    if env.load_state:
        traci.simulation.loadState(env.state_file)
        #####Potential Collision#######       
        if len(sumo_edges)==1 and modelType!="static":
            computeSafetyMeasure(traci,env,coShare_ForSafetyCheck,pedLaneWidth,bikeLaneWidth,carLaneWidth)
            
        if env.state_file not in env.generatedFiles:
            env.generatedFiles.append(env.state_file)

