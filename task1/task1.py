
#Task1 
#Aim: To predicts the percentage score by a student based on the no. of study hours

#importing Required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression as lg
from sklearn import metrics


#Loading the data 
url = "http://bit.ly/w-data"
data = pd.read_csv(url,thousands=",")
print(data.shape)
print(data.info())
print(data.describe())
print(data)
 

# Plotting the data on 2d graph
data.plot(x='Hours', y='Scores', style='o')  #plot the scatter Diagram
plt.title('Hours vs Percentage')             #Display the title of a graph
plt.xlabel('Hours Studied')                  #Display the X axis label
plt.ylabel('Percentage Score')               #Display the Y axis label
plt.show()                                   #Display the Scatter diagram


X = data.iloc[:,:-1].values                  #Split the data into features and labels
y = data.iloc[:, 1].values  
print(f" X= {X}")
print(f" y={y}")


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)  #Spilts the data into taining and test sets



regressor = lg()                             #Creates a object of linear regression class
regressor.fit(X_train, y_train)              #fits The data

print("Training complete.")



# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line)
plt.show()




print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores



df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df) 


hours =[[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 





#Result: The student will score 93.69% of marks if he/she studies for 9 hours in a day.