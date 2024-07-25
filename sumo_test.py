
from sumolib import checkBinary
import traci
import numpy as np
# cmd = ("/Users/damian/sumo/bin/sumo -c gym_sumo/envs/sumo_configs/intersection.sumocfg"
#        " --time-to-teleport.disconnected 1 --ignore-route-errors --device.rerouting.probability"
#        " 1 --device.rerouting.period 1 --pedestrian.striping.dawdling 0.5 --collision.check-junctions"
#        " True --collision.mingap-factor 0 --collision.action warn --random -W --default.carfollowmodel IDM --no-step-log --statistic-output output.xml")



SX = []
SY = []
test_load = True
for i, withGUI in enumerate([1,0]):
    sumoCMD = ["--end", "100", "--time-to-teleport.disconnected", str(1), "--ignore-route-errors", "--device.rerouting.probability", "1", "--device.rerouting.period", "1",
            "--pedestrian.striping.dawdling", "0.5", "--collision.check-junctions", "--collision.mingap-factor", "0", "--collision.action", "warn",
            "-W", "--default.carfollowmodel", "IDM", "--no-step-log", "--statistic-output", "output.xml"]

    sx = []
    sy = []
    if withGUI:
        sumoBinary = checkBinary('sumo-gui')
        sumoCMD += ["--start", "--quit-on-end"]
    else:
        sumoBinary = checkBinary('sumo')
    if i==2:
        import libsumo as traci


    _routeFileName = "environment/intersection_Slot_" + str(9) + ".rou.xml"
        
    sumoConfig = "gym_sumo/envs/sumo_configs/intersection.sumocfg"
    traci.start([sumoBinary] + ["-c", sumoConfig] + sumoCMD + ["--fcd-output", f"fcd{withGUI}.xml"] )
    
    if test_load:
        traci.simulationStep(100)
        traci.load(sumoCMD + ['-n', 'environment/intersection.net.xml', '-r', _routeFileName])
        # traci.load(sumoCMD + ["-c", sumoConfig] + ['-n', 'environment/intersection.net.xml', '-r', _routeFileName])

    for s in range(100):
        traci.simulationStep()


        laneID='E0_0'
        allVehicles = traci.lane.getLastStepVehicleIDs(laneID)			
        for veh in allVehicles:
            pos_bike_x,pos_bike_y = traci.vehicle.getPosition(veh)
            sx.append(pos_bike_x)
            sy.append(pos_bike_y)
    SX.append(sx)
    SY.append(sy)
    traci.close(False)
print(np.subtract(SX[0], SX[1]).sum())
# print(np.subtract(SX[1], SX[2]).sum())
print(np.subtract(SY[0], SY[1]).sum())
# print(np.subtract(SY[1], SY[2]).sum())