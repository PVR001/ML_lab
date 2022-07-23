import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy.linalg import inv,lstsq
from numpy import dot, transpose
df = pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv")
x=pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv",usecols=['PEER_PRESSURE'])
y=pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv",usecols=['SMOKING'])


model = LinearRegression()
model.fit(x,y)
x1=pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv",usecols=['PEER_PRESSURE'])
y1=pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv",usecols=['SMOKING'])

predictions = model.predict([[1]])
print("values of Predictions:",predictions)
print("values of β1, β2:",lstsq(x, y, rcond=None)[0])
print ("Score =",model.score(x1, y1))