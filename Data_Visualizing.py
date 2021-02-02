import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

Data_Frame = pd.read_csv('./Data/heart.csv')
theta_vector = pd.read_csv('./results/results.csv')
DataSet = np.array(Data_Frame)
theta = np.array(theta_vector)
features_matrix = DataSet[:, 0:-1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

pos = Data_Frame.loc[Data_Frame['target'] == 1]
neg = Data_Frame.loc[Data_Frame['target'] == 0]

pos_x = pos.iloc[:,3]
pos_y = pos.iloc[:,4]
pos_z = pos.iloc[:,0]

neg_x = neg.iloc[:,3]
neg_y = neg.iloc[:,4]
neg_z = neg.iloc[:,0]

ax.scatter(pos_x, pos_y, pos_z, marker='o')
ax.scatter(neg_x, neg_y, neg_z, marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()