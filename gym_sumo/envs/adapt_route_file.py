import xml.etree.ElementTree as ET
import numpy as np

np.random.seed(42)

def cointoss():
    return random.choice(["Heads", "Tails"])

#function
def adaptRouteFile(slotId, pid):

    routeFileName = f"environment/intersection_Slot_{slotId}_{pid}.rou.xml"   
    tree = ET.parse(routeFileName)
    root = tree.getroot()

    #change the count by +-10% for each randomly
    
    for flows in root.iter('flow'):
        if flows.attrib['id'] == "f_0":
            currentPedCount = int(flows.attrib['vehsPerHour'])
            percentage = np.random.randint(1, 5)
            if cointoss() == "Heads":
                currentPedCount = currentPedCount + int(currentPedCount*percentage/100)
            else:
                currentPedCount = currentPedCount - int(currentPedCount*percentage/100)
            flows.attrib['vehsPerHour'] = repr(currentPedCount)

        if flows.attrib['id'] == "f_1":
            bikeFlowCount = int(flows.attrib['vehsPerHour'])
            percentage = np.random.randint(1, 5)
            if cointoss() == "Heads":
                bikeFlowCount = bikeFlowCount + int(bikeFlowCount*percentage/100)
            else:
                bikeFlowCount = bikeFlowCount - int(bikeFlowCount*percentage/100)
            flows.attrib['vehsPerHour'] = repr(bikeFlowCount)

        if flows.attrib['id'] == "f_2":
            carFlowCount = int(flows.attrib['vehsPerHour'])
            percentage = np.random.randint(1, 5)
            if cointoss() == "Heads":
                carFlowCount = carFlowCount + int(carFlowCount*percentage/100)
            else:
                carFlowCount = carFlowCount - int(carFlowCount*percentage/100)
            flows.attrib['vehsPerHour'] = repr(carFlowCount)
    
    # filename = "environment/intersection_Slot_" + str(slotId) + ".rou.xml"
    file_handle = open(routeFileName,"wb")
    tree.write(file_handle)
    file_handle.close()
    print(currentPedCount)
    print(bikeFlowCount)
    print(carFlowCount)
