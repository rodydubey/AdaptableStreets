# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib as mpl
# import numpy as np
# import seaborn as sns


# #load data
# dataFromCSV = pd.read_csv('stat_train_for_visualization.csv', sep=',')
# print(dataFromCSV.shape)

# fig = plt.figure(figsize=(8, 6))
# t = fig.suptitle('Level of Service and independent variables', fontsize=14)
# ax = fig.add_subplot(111, projection='3d')

# xs = list(dataFromCSV['LevelOfService'])
# ys = list(dataFromCSV['Ped_Lane_Width'])
# zs = list(dataFromCSV['Total_occupancy_ped_Lane'])
# data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]

# ss = list(dataFromCSV['Hinderance_bb'])
# colors = ['red' if wt == 'red' else 'yellow' for wt in list(dataFromCSV['Hinderance_pp'])]
# markers = [',' if q == 'high' else 'x' if q == 'medium' else 'o' for q in list(dataFromCSV['Hinderance_bp'])]

# for data, color, size, mark in zip(data_points, colors, ss, markers):
#     x, y, z = data
#     ax.scatter(x, y, z, alpha=0.4, c=color, edgecolors='none', s=size, marker=mark)

# ax.set_xlabel('LevelOfService')
# ax.set_ylabel('Ped_Lane_Width')
# ax.set_zlabel('Total_occupancy_ped_Lane')


import pandas as pd
import plotly
import plotly.graph_objs as go


#Read cars data from csv
data = pd.read_csv("stat_train.csv")

#Set marker properties
markercolor = data['Ped_Lane_Width']
markersize = data['Hinderance_bp']

#Make Plotly figure
fig1 = go.Scatter3d(x=data['LevelOfService'],
                    y=data['Total_occupancy_ped_Lane'],
                    z=data['Hinderance_pp'],
                   marker=dict(size=markersize,
                                color=markercolor,
                                opacity=0.9,
                                reversescale=True,
                                colorscale='reds'),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="LevelOfService"),
                                yaxis=dict( title="Total_occupancy_ped_Lane"),
                                zaxis=dict(title="Hinderance_pp")),)

#Plot and save html
plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("5DPlot.html"))

# #Set marker properties
# markersize = data['Hinderance_bp']
# markercolor = data['Hinderance_pp']
# markershape = data['Hinderance_bb'].replace("low","square").replace("high","circle")


# #Make Plotly figure
# fig1 = go.Scatter3d(x=data['LevelOfService'],
#                     y=data['Ped_Lane_Width'],
#                     z=data['Total_occupancy_ped_Lane'],
#                     marker=dict(size=markersize,
#                                 color=markercolor,
#                                 symbol=markershape,
#                                 opacity=1.0,
#                                 reversescale=True,
#                                 colorscale='Blues'),
#                     line=dict (width=0.02),
#                     mode='markers')

# #Make Plot.ly Layout
# mylayout = go.Layout(scene=dict(xaxis=dict( title="LevelOfService"),
#                                 yaxis=dict( title="Ped_Lane_Width"),
#                                 zaxis=dict(title="Total_occupancy_ped_Lane")),)

# #Plot and save html
# plotly.offline.plot({"data": [fig1],
#                      "layout": mylayout},
#                      auto_open=True,
#                      filename=("6DPlot.html"))

