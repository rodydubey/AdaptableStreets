from lxml import etree
import os
import sys
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

np.random.seed(42)
root = etree.Element("routes")

from_tos = [('E0', 'E2'), ("-E1", "E3"), ("-E2", "-E0"), ("-E3", "E1")]
veh_types = {'car': {'type': 'DEFAULT_VEHTYPE',
                     'color': 'yellow',
                     'id': 'f_2'},
             'ped': {'type': 'DEFAULT_PEDTYPE',
                     'color': 'magenta',
                     'id': 'f_0'},
             'bike': {'type': 'DEFAULT_BIKETYPE',
                     'color': 'blue',
                     'id': 'f_1'}
                     }

import csv
i = 1
# edges = ['E0', '-E1', '-E2', '-E3']
edges = ['E0']
with open("trafficFlow.csv", 'r', encoding='utf-8-sig') as file:
  csvreader = csv.reader(file)
  for i, row in enumerate(csvreader):
    for (edge_id, to_edge) in from_tos:
      if edge_id not in edges:
        continue
      for (flowcount, fid, attribs) in zip(row, [0,1,2], veh_types.values()): #[car, bike, ped]
        for j in range(6): #action steps
            flows = etree.SubElement(root, "flow", attrib=attribs)
            
            flows.attrib['id'] = f"{edge_id}_slot_{i}_action_{j}_f_{fid}"
            flows.attrib['vehsPerHour'] = str(flowcount)
            flows.attrib['from'] = edge_id
            flows.attrib['to'] = to_edge
            flows.attrib['begin'] = str(i*1800 + j*300)
            flows.attrib['end'] = str(i*1800 + j*300 + 300)
filename = "testcase_0/daytest/flows.rou.xml"
file_handle = open(filename,"wb")
tree = etree.ElementTree(root)
tree.write(file_handle, pretty_print=True)
file_handle.close()

i = 1
tree = etree.parse('environment/intersection_Slot_3_backup.rou.xml')
root = tree.getroot()
with open("trafficFlow.csv", 'r', encoding='utf-8-sig') as file:
  csvreader = csv.reader(file)
  for j, row in enumerate(csvreader):
    for flows in root.iter('flow'):
        edge_id = flows.attrib['id'].split('_')[0]
        if edge_id not in edges:
          flows.getparent().remove(flows)
        else:
          if flows.attrib['id'] == f"{edge_id}_f_0":
              pedFlowCount = row[2]
              factor = 1
              # if j>27 and j<=32:
              #    factor = 1.5
              # if j>31 and j<=38:
              #   factor = 1.5
              pedFlowCount = str(float(pedFlowCount)*factor)
              flows.attrib['vehsPerHour'] = pedFlowCount

          if flows.attrib['id'] == f"{edge_id}_f_1":
              bikeFlowCount = row[1]
              factor = 1
              # if i>27 and i<=32:
              #    factor = 1.5
              # if i>31 and i<=38:
              #   factor = 1.5
              bikeFlowCount = str(float(bikeFlowCount)*factor)
              flows.attrib['vehsPerHour'] = bikeFlowCount

          if flows.attrib['id'] == f"{edge_id}_f_2":
              carFlowCount = row[0]
              factor = 1
              # if i>27 and i<=32:
              #    factor = 0.8
              # if i>31 and i<=38:
              #     factor = 1.5
              carFlowCount = str(float(carFlowCount)*factor)
              flows.attrib['vehsPerHour'] = carFlowCount
    
    if len(edges)==4:
       foldername = '4way'
    else:
       foldername = 'two'
    filename = f"testcase_0/{foldername}/intersection_Slot_" + str(i) + ".rou.xml"
    file_handle = open(filename,"wb")
    tree.write(file_handle)
    file_handle.close()
    i+=1
