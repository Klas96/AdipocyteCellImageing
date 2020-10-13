__author__ = "Klas Holmgren"
__license__ = "Feel free to copy"

#Imports
import numpy as np
#Set the `numpy` pseudo-random generator at a fixed value
#This helps with repeatable results everytime you run the code.
np.random.seed(42)
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import linear_model
from LoadData import loadIMGsAsDataFrame

#import data as pandas data frame

path  = 'images_for_preview/60x images/input/'
inputDf, inputShape = loadIMGsAsDataFrame(path)

path  = 'images_for_preview/60x images/targets/'
targetDf, targetShape = loadIMGsAsDataFrame(path)

#Create Linear Regression object
reg = linear_model.LinearRegression()


#Fit Regression Model
reg.fit(inputDf, targetDf.Param1)

#Print coefficients
#print(reg.coef_, reg.intercept_)

#Test Prediction
path = 'images_for_preview/40x images/input/'
testDf, testShape = loadIMGsAsDataFrame(path)

pred = reg.predict(testDf)

print(pred)

pred = pred.reshape(targetShape)

plt.imshow(pred)
plt.show()
