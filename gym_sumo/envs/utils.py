import xml.etree.ElementTree as ET
import os
import sys
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

tree = ET.parse('environment\intersection_Slot_3_backup.rou.xml')
root = tree.getroot()


# carflowLevelDict = {'l':100,'m':700,'h':1250}
# bikeflowLevelDict = {'l':22,'m':63,'h':96}
# pedflowLevelDict = {'l':15,'m':122,'h':250}

carflowLevelDict = {'l':200,'m':600,'h':1200}
bikeflowLevelDict = {'l':200,'m':600,'h':1200}
pedflowLevelDict = {'l':200,'m':600,'h':1200}

#dictionary of index with traffic flow data first (ped), second(bike), third (car)
trafficFlowDataAll = {'1':'l_l_l','2':'l_l_m','3':'l_l_h','4':'l_m_l','5':'l_m_m','6':'l_l_h','7':'l_h_l','8':'l_h_m','9':'l_h_h','10':'m_l_m','11':'m_l_m'
                        ,'12':'m_l_h','13':'m_m_l','14':'m_m_m','15':'m_m_h','16':'m_h_l','17':'m_h_m','18':'m_h_h','19':'h_l_l','20':'h_l_m','21':'h_l_h','22':'h_m_l'
                        ,'23':'h_m_m','24':'h_m_h','25':'h_h_l','26':'h_h_m','27':'h_h_h'}


def generateFlowFiles(scenario):

    if scenario=="Test 0":

        for i in range(288):
            
            #pick a random integer between 1 to 27
            randomIndex = random.randint(1,27)
            combination = trafficFlowDataAll.get(str(randomIndex))
            x = combination.split("_",2)
            pedFlow = x[0]
            bikeFlow = x[1]
            carFlow = x[2]
            
            pedFlowCount =  pedflowLevelDict.get(str(pedFlow)) + random.randint(-100,+100)
            bikeFlowCount =  bikeflowLevelDict.get(str(bikeFlow)) + random.randint(-100,+100)
            carFlowCount =  carflowLevelDict.get(str(carFlow)) + random.randint(-100,+100)

            for flows in root.iter('flow'):
                if flows.attrib['id'] == "f_0":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_3":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_6":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                
                if flows.attrib['id'] == "f_9":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_12":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_15":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)

                if flows.attrib['id'] == "f_18":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_21":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_24":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)

                if flows.attrib['id'] == "f_27":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_30":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_33":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)


                if flows.attrib['id'] == "f_1":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_4":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_7":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                
                if flows.attrib['id'] == "f_10":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_13":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_16":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)

                if flows.attrib['id'] == "f_19":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_22":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_25":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)

                if flows.attrib['id'] == "f_28":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_31":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_34":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)

                
                if flows.attrib['id'] == "f_2":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_5":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_8":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                
                if flows.attrib['id'] == "f_11":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_14":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_17":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)

                if flows.attrib['id'] == "f_20":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_23":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_26":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)

                if flows.attrib['id'] == "f_29":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_32":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_35":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)

            filename = "testcase_0\intersection_Slot_" + str(i+1) + ".rou.xml"
            file_handle = open(filename,"wb")
            tree.write(file_handle)
            file_handle.close()

    elif scenario=="Test 1":
        for i in range(288):
            
            #pick a random integer between 1 to 27
            randomIndex = random.randint(1,27)
            combination = trafficFlowDataAll.get(str(randomIndex))
            x = combination.split("_",2)
            pedFlow = x[0]
            bikeFlow = x[1]
            carFlow = x[2]
            
            pedFlowCount =  pedflowLevelDict.get(str(pedFlow)) + random.randint(-100,+100)
            bikeFlowCount =  bikeflowLevelDict.get(str(bikeFlow)) + random.randint(-100,+100)
            carFlowCount =  carflowLevelDict.get(str(carFlow)) + random.randint(-100,+100)

            for flows in root.iter('flow'):
                if flows.attrib['id'] == "f_0":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_3":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_6":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                
                if flows.attrib['id'] == "f_9":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_12":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_15":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)

                if flows.attrib['id'] == "f_18":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_21":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_24":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)

                if flows.attrib['id'] == "f_27":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_30":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_33":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)


                if flows.attrib['id'] == "f_1":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_4":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_7":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                
                if flows.attrib['id'] == "f_10":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_13":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_16":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)

                if flows.attrib['id'] == "f_19":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_22":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_25":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)

                if flows.attrib['id'] == "f_28":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_31":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_34":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)

                
                if flows.attrib['id'] == "f_2":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_5":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_8":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                
                if flows.attrib['id'] == "f_11":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_14":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_17":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)

                if flows.attrib['id'] == "f_20":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_23":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_26":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)

                if flows.attrib['id'] == "f_29":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_32":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_35":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)

            filename = "testcase_1\intersection_Slot_" + str(i+1) + ".rou.xml"
            file_handle = open(filename,"wb")
            tree.write(file_handle)
            file_handle.close()
    else:
        for i in range(27):            
            combination = trafficFlowDataAll.get(str(i+1))
            x = combination.split("_",2)
            pedFlow = x[0]
            bikeFlow = x[1]
            carFlow = x[2]

            pedFlowCount =  pedflowLevelDict.get(str(pedFlow)) 
            bikeFlowCount =  bikeflowLevelDict.get(str(bikeFlow))
            carFlowCount =  carflowLevelDict.get(str(carFlow)) 

            for flows in root.iter('flow'):
                if flows.attrib['id'] == "f_0":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_3":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_6":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                
                if flows.attrib['id'] == "f_9":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_12":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_15":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)

                if flows.attrib['id'] == "f_18":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_21":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_24":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)

                if flows.attrib['id'] == "f_27":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_30":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)
                if flows.attrib['id'] == "f_33":
                    flows.attrib['vehsPerHour'] = repr(pedFlowCount)


                if flows.attrib['id'] == "f_1":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_4":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_7":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                
                if flows.attrib['id'] == "f_10":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_13":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_16":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)

                if flows.attrib['id'] == "f_19":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_22":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_25":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)

                if flows.attrib['id'] == "f_28":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_31":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)
                if flows.attrib['id'] == "f_34":
                    flows.attrib['vehsPerHour'] = repr(bikeFlowCount)

                
                if flows.attrib['id'] == "f_2":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_5":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_8":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                
                if flows.attrib['id'] == "f_11":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_14":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_17":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)

                if flows.attrib['id'] == "f_20":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_23":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_26":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)

                if flows.attrib['id'] == "f_29":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_32":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)
                if flows.attrib['id'] == "f_35":
                    flows.attrib['vehsPerHour'] = repr(carFlowCount)

            filename = "environment\intersection_Slot_" + str(i+1) + ".rou.xml"
            file_handle = open(filename,"wb")
            tree.write(file_handle)
            file_handle.close()

def gather(env_info):
    rewards = env_info.rewards
    next_states = env_info.vector_observations
    dones = env_info.local_done
    return rewards, next_states, dones   

def print_status(episode, scores, total_scores):
    avg100 = np.mean(np.array(total_scores).T[0][-100:])
    # distance_info = f"dis: {np.mean(agent.distances[-1000:]):.3f} sclr: {agent.scalar:.5f}" if agent.noise_type == "param" else "" 
    print(f"Ep {episode}\tAvg100: {avg100:.2f}\tMean (min|max): {np.mean(scores):.2f} ({np.min(scores):.2f}|{np.max(scores):.2f})")


def plot_scores(scores_array, labels, save_as=None):
    fig = plt.figure(figsize=(12,7))

    for scores, label in zip(scores_array, labels):
        scores = np.array(scores)
        if scores.ndim > 1:
            transposed = scores.T
            plt.plot(np.arange(1, len(scores)+1), transposed[0], label=label)
            plt.fill_between(np.arange(1, len(scores)+1), transposed[1], transposed[2], alpha=0.2)
        else:
            plt.plot(np.arange(1, len(scores)+1), scores, label=label)

    # plt.axhline(y=30, color='green', linestyle='dashed')
    # plt.axhline(y=37, color='lightgray', linestyle='dashed')
    # plt.axhline(y=40, color='lightgray')

    plt.legend(loc='lower right')

    plt.ylabel('Score')
    plt.xlabel('Episode #')

    if save_as:
        plt.savefig(save_as)

    plt.show()