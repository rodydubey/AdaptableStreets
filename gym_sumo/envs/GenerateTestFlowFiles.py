import xml.etree.ElementTree as ET
import os
import sys
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

np.random.seed(42)
tree = ET.parse('environment/intersection_Slot_3_backup.rou.xml')
root = tree.getroot()

import csv
i = 1
edges = ['E0', '-E1', '-E2', '-E3']
with open("trafficFlow.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    for flows in root.iter('flow'):
      for edge_id in edges:
        if flows.attrib['id'] == f"{edge_id}_f_0":
            pedFlowCount = row[2]
            flows.attrib['vehsPerHour'] = pedFlowCount

        if flows.attrib['id'] == f"{edge_id}_f_1":
            bikeFlowCount = row[1]
            flows.attrib['vehsPerHour'] = bikeFlowCount

        if flows.attrib['id'] == f"{edge_id}_f_2":
            carFlowCount = row[0]
            flows.attrib['vehsPerHour'] = carFlowCount
    
   
    filename = "testcase_0/4way/intersection_Slot_" + str(i) + ".rou.xml"
    file_handle = open(filename,"wb")
    tree.write(file_handle)
    file_handle.close()
    i+=1
