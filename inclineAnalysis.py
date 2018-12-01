
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import reduce,partial
from iminuit import Minuit
from probfit import Chi2Regression
from scipy import stats





filename = "inclineTime1.txt"




delimiter = "\t"

# We import the data as a pandas dataframe, then print it for good measure.
data = pd.read_csv(filename,sep = delimiter,names = ["Time","Voltage"])
dataValues = data.values
t = dataValues[:,0]
volt = dataValues[:,1]

print(data)


dataPlot = sns.lineplot(x = "Time",y = "Voltage",data = data)

plt.show()

points = []

inside = False

threshold = 1.0

for i in range(1,len(t)):
    if (volt[i] > threshold and not inside):
        inside = True
        points.append([i,t[i],volt[i]])
    if (volt[i] <= threshold and inside):
        inside = False
    
        
print(points)
