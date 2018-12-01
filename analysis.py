"""
Preconfigured analysis functions for the friday experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import reduce,partial
from iminuit import Minuit
from probfit import Chi2Regression
from scipy import stats


"""
________SECTION 1_______________
Fitting of time wrt. the pendulum experiment 





"""





"""
Importing and viewing
"""

# The name of the file we want to import
filename = "Anders1.dat"

# Delimiter
delimiter = "\t"

# We import the data as a pandas dataframe, then print it for good measure.
data = pd.read_csv(filename,sep = delimiter,names = ["Indexes","Measurements"])
print(data)

# We retrive an numpy array with the values
dataMatrix = data.values


#dataPlot = sns.scatterplot(x = "Indexes",y = "Measurements",data = data)
#fig1, ax1 = plt.subplots()
#dataPlot = sns.relplot(x = "Indexes",y = "Measurements",data = data,kind = "scatter")




"""
Function fiting
"""


#Notice it has no offset.
#FIXME: Trying with offset
def linearFitFunction(x,a,b):
    return a*x + b


#We want to find the best fit using rms, since we don't know the errors (yet)
#This is equvialent to using χ² with σᵢ= 1 ∀i.
np.ones_like(data.values[:,0])
lsLinearTimeReg = Chi2Regression(linearFitFunction,
                              data.values[:,0],
                              data.values[:,1],
                              np.ones_like(data.values[:,0]))

a_start = 7.0
miniutFit = Minuit(lsLinearTimeReg,pedantic = False, a = a_start, b = 0)
miniutFit.migrad();
ls_a = miniutFit.args[0]
ls_b = miniutFit.args[1]

fittedLinear = partial(linearFitFunction,a = ls_a,b = ls_b)

residuals = list(map((lambda x: fittedLinear(x[0]) - x[1]),dataMatrix))


ndfData =  np.transpose([range(0,len(residuals)),residuals])

ndf = pd.DataFrame(data = ndfData,columns = ["x-val","Residuals"])

#newDataPlot = sns.relplot(x = "x-val",y = "Residuals",data = ndf,kind = "scatter")

#plt.show()




#sns.pointplot(x = "x-val",y = "Residuals",data = ndf,ci = resStd,join = False)
#plt.errorbar(data.values[:,0],data.values[:,1], resStd*np.ones_like(residuals))

#plt.show()




#We want to plot an histogram of the errorss
y,x_edges = np.histogram(residuals,bins = 10)

x_center = x_edges[:-1] + (x_edges[1]-x_edges[0])/2

plt.plot(x_center,y)
plt.show()




resMean = np.mean(residuals)
resStd = np.std(residuals)


print("Mean of the residuals:",resMean)
print("Std of the residuals:",resStd)

# Assuming the results are gausian, we can redo the fit:
#FIXME: Not sure if this is correct. See video for info.
#might instead be resStd/(math.sqrt(len(residuals))) ?

chi2Line = Chi2Regression(linearFitFunction,
                          data.values[:,0],
                          data.values[:,1],
                          resStd*np.ones_like(data.values[:,0]))

chi2miniutFit = Minuit(chi2Line,pedantic = False, a = ls_a, b = ls_b)
chi2miniutFit.migrad();
chi2Testimate = miniutFit.args[0]




#

"""
Step sizes
"""



# measurements = [0] + list(dataMatrix[:,1])

# steps = np.zeros(len(measurements)- 1)
# val = 0
# for i in range(0,len(measurements) - 1):
#     steps[i] = measurements[i + 1] - measurements[i]

# print(steps)

# ndfData =  np.transpose([range(0,len(steps)),steps])

# ndf = pd.DataFrame(data = ndfData,columns = ["Indexes","Measurements"])

# newDataPlot = sns.relplot(x = "Indexes",y = "Measurements",data = ndf,kind = "scatter")

# plt.show()




#As always, we want to plot our data:








