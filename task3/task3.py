#Importing Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#loading the data
dir = os.getcwd()
url = os.path.join(dir,".vscode\\task3\\data.csv")
data = pd.read_csv(url)


#printing some rows 
print(data.shape)
print(data.dtypes)
print(data.head())
print(data.tail())
print(data.describe())
print(type(data))
print(data.nunique())


#cleaning the data
print(data.isnull().sum())

#Checking for outliers
sns.boxplot(x=data["Sales"])
plt.show()
sns.boxplot(x=data["Quantity"])
plt.show()
sns.boxplot(x=data["Discount"])
plt.show()
sns.boxplot(x=data["Profit"])
plt.show()

#inter quatile range

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

#histogram
data.hist(figsize=(40,50))
plt.show()



# Finding the relations between the variables.
plt.figure(figsize=(20,10))
c= data.corr()
sns.heatmap(c,cmap = "BrBG" , annot=True)
c
plt.show()


# Plotting a scatter plot
fig, ax = plt.subplots(figsize=(10,6))
plt.scatter(data["Sales"],data["Profit"])
ax.set_xlabel("Sales")
ax.set_ylabel("Profit")
plt.show()

