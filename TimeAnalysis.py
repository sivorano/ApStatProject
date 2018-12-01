import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import reduce,partial
from iminuit import Minuit
from probfit import Chi2Regression
from scipy import stats



filenames = ["Anders1.dat",
             #FIXME: Annas data need massaging before it is usefull"Anna"
             "Kerttu1.dat",
             "Søren3.dat"]


# Delimiter
delimiter = "\t"


dataListDf = []
dataListMatrix = []

for name in filenames:
    dataTemp = pd.read_csv(name,sep = delimiter,names = ["Indexes","Measurements"])
    dataListDf.append(dataTemp)
    dataListMatrix.append(dataTemp.values)



"""
Fitting
"""
def linearFitFunction(x,a,b):
    return a*x + b


for data in dataListDf:

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
    
    residuals = list(map((lambda x: fittedLinear(x[0]) - x[1]),data.values))
    
    y,x_edges = np.histogram(residuals,bins = 10)

    x_center = x_edges[:-1] + (x_edges[1]-x_edges[0])/2

    plt.plot(x_center,y)
    plt.show()




    resMean = np.mean(residuals)
    resStd = np.std(residuals)

    print("Mean of the residuals:",resMean)
    print("Std of the residuals:",resStd)


"""
Combined data
"""


adjDataList = []
collectedData = np.array([])
for data in dataListMatrix:
    adjData = data - data[0] #They now start from time 0
    adjDataList.append(adjData)
    np.append(collectedData,adjData)

colData = reduce((lambda x,y : np.append(x,y,axis= 0)),adjDataList)

colDf =  pd.DataFrame(data = colData,columns = ["Indexes","Measurements"])



