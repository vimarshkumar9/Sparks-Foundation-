import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn import metrics

#loading the data 
iris_set = datasets.load_iris()
iris_data = pd.DataFrame(iris_set.data, columns=iris_set.feature_names)
iris_data["species"] = iris_set.target
iris_data.head()
iris_data.shape


#how many data points for each class are present?
iris_data["species"].value_counts()

#checking for null values
iris_data.isnull().sum()

#scatterplot to check for relationship between sepal length and sepal width
sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))
plt.title('Sepal Dimensions')
sns.scatterplot(x=iris_data["sepal length (cm)"], y=iris_data["sepal width (cm)"], hue=iris_data["species"],palette = ["green","orange","blue"],s=100)
plt.show()

#scatterplot to check for relationship between petal length and petal width
plt.figure(figsize=(12, 6))
plt.title('Petal Dimensions')
sns.scatterplot(x=iris_data["petal length (cm)"],y=iris_data["petal width (cm)"], hue=iris_data["species"],palette = ["green","orange","blue"],s=100)
plt.show()


#plotting histograms for distributions of sepal length ,width and petal width and length
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

#plot for sepal length
axes[0,0].set_title('Distribution of Sepal Length')
axes[0,0].hist(iris_data["sepal length (cm)"])


#plot for sepal width
axes[0,1].set_title('Distribution of Sepal Width')
axes[0,1].hist(iris_data["sepal width (cm)"])

#plot for petal length
axes[1,0].set_title('Distribution of Petal Length')
axes[1,0].hist(iris_data["petal length (cm)"]) 

#plot for petal width
axes[1,1].set_title('Distribution of Petal Width')
axes[1,1].hist(iris_data["petal width (cm)"])
plt.show()

#plotting correaltion heatmap

plt.figure(figsize=(8,4))
sns.heatmap(iris_data.corr(), annot=True, cmap='Blues')
plt.show()

#spilting the dataset and training it 
train, test = train_test_split(iris_data, test_size = 0.2)
train.shape, test.shape
train_x = train[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]
train_y = train.species
test_x = test[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]
test_y = test.species


#defining the decision tree classifier
dtree = DecisionTreeClassifier()
dtree.fit(train_x,train_y)
predictions = dtree.predict(test_x)
print("The accuracy of Decision Tree is:", metrics.accuracy_score(predictions, test_y))
X = iris_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]
Y = iris_data.species

dtree1 = DecisionTreeClassifier()
dtree1.fit(X,Y)
print("Decision Tree Classifier is created")


#plotting the tree graph
dot_data = export_graphviz(dtree,out_file=None,feature_names=iris_set.feature_names,class_names=iris_set.target_names,filled=True,rounded=True,special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")

