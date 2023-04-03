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
# i = 1
# with open("trafficFlow.csv", 'r') as file:
#   csvreader = csv.reader(file)
#   for row in csvreader:
#     for flows in root.iter('flow'):
#         if flows.attrib['id'] == "f_0":
#             pedFlowCount = row[2]
#             flows.attrib['vehsPerHour'] = pedFlowCount

#         if flows.attrib['id'] == "f_1":
#             bikeFlowCount = row[1]
#             flows.attrib['vehsPerHour'] = bikeFlowCount

#         if flows.attrib['id'] == "f_2":
#             carFlowCount = row[0]
#             flows.attrib['vehsPerHour'] = carFlowCount
    
   
#     filename = "testcase_0\one\intersection_Slot_" + str(i) + ".rou.xml"
#     file_handle = open(filename,"wb")
#     tree.write(file_handle)
#     file_handle.close()
#     i+=1
avg_waiting_time_car_temp = []
avg_waiting_time_bike_temp = []
avg_waiting_time_ped_temp = []
avg_queue_length_car_temp = []
avg_queue_length_bike_temp = []
avg_queue_length_ped_temp = []
los_temp = []
cosharing_temp = []
avg_waiting_time_car = []
avg_waiting_time_bike = []
avg_waiting_time_ped = []
avg_queue_length_car = []
avg_queue_length_bike = []
avg_queue_length_ped = []
los = []
cosharing = []
with open("results/static_test_surge_run74.csv", 'r') as file:
  csvreader = csv.reader(file)
  i = 0
  headings = next(csvreader)
  for row in csvreader:   
    if i % 6 == 0 and i!=0:
        avg_waiting_time_car.append(np.mean(avg_waiting_time_car_temp))
        avg_waiting_time_bike.append(np.mean(avg_waiting_time_bike_temp))
        avg_waiting_time_ped.append(np.mean(avg_waiting_time_ped_temp))
        avg_queue_length_car.append(np.mean(avg_queue_length_car_temp))
        avg_queue_length_bike.append(np.mean(avg_queue_length_bike_temp))
        avg_queue_length_ped.append(np.mean(avg_queue_length_ped_temp))
        los.append(np.mean(los_temp))
        cosharing.append(np.mean(cosharing_temp))
        avg_waiting_time_car_temp.clear()
        avg_waiting_time_bike_temp.clear()
        avg_waiting_time_ped_temp.clear()
        avg_queue_length_car_temp.clear()
        avg_queue_length_bike_temp.clear()
        avg_queue_length_ped_temp.clear()
        los_temp.clear()
        cosharing_temp.clear()
        avg_waiting_time_car_temp.append(float(row[0]))
        avg_waiting_time_bike_temp.append(float(row[1]))
        avg_waiting_time_ped_temp.append(float(row[2]))
        avg_queue_length_car_temp.append(float(row[3]))
        avg_queue_length_bike_temp.append(float(row[4]))
        avg_queue_length_ped_temp.append(float(row[5]))
        los_temp.append(float(row[6]))
        if row[8] == "True":
          cosharing_temp.append(float(1))
        else:
          cosharing_temp.append(float(0))   
        i+=1
    else:
        avg_waiting_time_car_temp.append(float(row[0]))
        avg_waiting_time_bike_temp.append(float(row[1]))
        avg_waiting_time_ped_temp.append(float(row[2]))
        avg_queue_length_car_temp.append(float(row[3]))
        avg_queue_length_bike_temp.append(float(row[4]))
        avg_queue_length_ped_temp.append(float(row[5]))
        los_temp.append(float(row[6]))   
        if row[8] == "True":
          cosharing_temp.append(float(1))
        else:
          cosharing_temp.append(float(0))     
        i+=1
avg_waiting_time_car.append(np.mean(avg_waiting_time_car_temp))
avg_waiting_time_bike.append(np.mean(avg_waiting_time_bike_temp))
avg_waiting_time_ped.append(np.mean(avg_waiting_time_ped_temp))
avg_queue_length_car.append(np.mean(avg_queue_length_car_temp))
avg_queue_length_bike.append(np.mean(avg_queue_length_bike_temp))
avg_queue_length_ped.append(np.mean(avg_queue_length_ped_temp))
los.append(np.mean(los_temp))
cosharing.append(np.mean(cosharing_temp))
newFileName = "results/static_test_surge_run74_revised.csv"
with open(newFileName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['avg_waiting_time_car','avg_waiting_time_bike','avg_waiting_time_ped','avg_queue_length_car','avg_queue_length_bike','avg_queue_length_ped','los','cosharing'])
        i = 0
        for i in range(len(avg_waiting_time_car)):
            writer.writerow([avg_waiting_time_car[i],avg_waiting_time_bike[i],avg_waiting_time_ped[i],avg_queue_length_car[i],avg_queue_length_bike[i],avg_queue_length_ped[i],los[i],cosharing[i]])
           


        
