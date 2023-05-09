from lxml import etree
import os
import numpy as np


CAR = 2
BIKE = 1
PED = 0

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

def singleFlowFileTest():
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


def generateFlowFiles(edges,  df, surge_factors={}, surge_timeslots=[], noise=False):
    i = 1
    tree = etree.parse('environment/base_flow.rou.xml')
    root = tree.getroot()

    for j, row in df.iterrows():
        for flows in root.iter('flow'):
            edge_id = flows.attrib['id'].split('_')[0]
            if edge_id not in edges:
                flows.getparent().remove(flows)
            else:
                for veh_type, key in veh_keys.items():
                    if flows.attrib['id'] == f"{edge_id}_f_{veh_type}":
                        if j in surge_timeslots:
                            surge_factor = surge_factors.get(veh_type, 1)
                        else:
                            surge_factor = 1
                        FlowCount = row[f'{key}_{edge_id}']
                        if noise:
                            surge_factor *= (np.random.randn()*0.05+1)
                        FlowCount = str(float(FlowCount)*surge_factor)
                        flows.attrib['vehsPerHour'] = FlowCount
        lane_arms = "single"
        surge_name = 'one'
        if len(edges) == 4:
            lane_arms = '4way'
        if surge_factors:
            surge_name = 'two'
        foldername = f"barcelona_test/{lane_arms}/{surge_name}"
        os.makedirs(foldername, exist_ok=True)
        filename = f"{foldername}/intersection_Slot_" + str(j+1) + ".rou.xml"
        file_handle = open(filename, "wb")
        tree.write(file_handle)
        file_handle.close()
        i += 1


if __name__ == "__main__":
    import pandas as pd
    edge_sets = [['E0'], ['E0', '-E1', '-E2', '-E3']]
    veh_keys = {CAR: 'vehs_sim',
                BIKE: 'bikes_real',
                PED: 'peds'}
    surge_factors = {CAR: 1.5,
                     BIKE: 5,
                     PED: 2}
    surge_timeslots = range(29,40) # timeslot 30 to 40
    df = pd.read_csv("trafficFlow_test.csv")
    for edges in edge_sets:
        for factor in surge_factors:
            generateFlowFiles(edges, df, surge_factors, surge_timeslots)

