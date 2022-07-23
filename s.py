# importing the module
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy.linalg import inv,lstsq
from numpy import dot, transpose

# read specific columns of csv file using Pandas
df = pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv")
x=pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv",usecols=['PEER_PRESSURE'])
y=pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv",usecols=['SMOKING'])

model = LinearRegression()
model.fit(x,y)
plt.figure()    
plt.title('LUNG CANCER(i) - influenced by smoking')
plt.xlabel('AGE')
plt.ylabel('no of people')
plt.plot(x,y,'.')
plt.plot(x,model.predict(x),'-')
plt.axis([0,25,0,25])
plt.grid(True)
print ("Predicted data =",model.predict([[21]])) # 22.467
plt.show()


x=pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv",usecols=['PEER_PRESSURE'])
y=pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv",usecols=['SMOKING'])
model = LinearRegression()

model.fit(x,y)
x_test=pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv",usecols=['PEER_PRESSURE'])
y_test=pd.read_csv("D:/mcalab02/Desktop/New folder/lung.csv",usecols=['SMOKING'])

print("SCORE=",model.score(x_test,y_test))