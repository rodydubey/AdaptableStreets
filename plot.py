# import pandas as pd
# from matplotlib import pyplot as plt

# # Set the figure size
# plt.rcParams["figure.figsize"] = [7.00, 5.00]
# plt.rcParams["figure.autolayout"] = True
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["r", "#e94cdc", "0.7"]) 
# # Make a list of columns
# columns = ['Adaptable', 'Static', 'Random']

# # Read a CSV file
# df = pd.read_csv("ForPeds_AllScenarios.csv", usecols=columns)

# # Plot the lines
# df.plot()

# plt.show()


# import pandas as pd
# from matplotlib import pyplot as plt
# import numpy as np
# # Set the figure size
# plt.rcParams["figure.figsize"] = [7.00, 5.00]
# plt.rcParams["figure.autolayout"] = True
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["r", "#e94cdc", "0.7"]) 
# # Make a list of columns
# columns = ['Human_CognitiveAgent', 'Human_RandomAgent', 'Human_HeuristicAgent']
# # labels for x-asix
# labels = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5','T6', 'T7', 'T8', 'T9', 'T10', 'T11','T12', 'T13', 'T14', 'T15', 'T16', 'T17']
# x = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5','T6', 'T7', 'T8', 'T9', 'T10', 'T11','T12', 'T13', 'T14', 'T15', 'T16', 'T17']
# # setting x-axis values
# # plt.xticks(x,labels)
# # naming of x-axis and y-axis
# plt.xlabel("All Zones (Floot T)")
# plt.ylabel("DTW")

# # Read a CSV file
# df = pd.read_csv("DTW.csv", usecols=columns)

# # Plot the lines
# df.plot()

# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
# reading CSV file
data = pd.read_csv("DTW_1.csv")
# Define data values
# x = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5','T6', 'T7', 'T8', 'T9', 'T10', 'T11','T12', 'T13', 'T14', 'T15', 'T16', 'T17']
# y = [5, 12, 19, 21, 31, 27, 35]
# z = [3, 5, 11, 20, 15, 29, 31]

# converting column data to list
x = data['Zone'].tolist()
y1 = data['Human_CognitiveAgent'].tolist()
y2 = data['Human_RandomAgent'].tolist()
y3 = data['Human_HeuristicAgent'].tolist()

plt.plot(x, y1, "-b", label="Human vs CognitiveAgent")
plt.plot(x, y2,  "-r", label="Human vs RandomAgent")
plt.plot(x, y3,  "-y", label="Human vs HeuristicAgent")
plt.legend(loc="upper left")
plt.xlabel("All Zones (Floor 1)")
plt.ylabel("DTW")

plt.show()