from sklearn import datasets
iris = datasets.load_iris()



import os

os.system("PAUSE")

from sklearn import tree
import pandas as pd
dftrain = pd.read_csv('training.csv')

training_labels = dftrain.iloc[:,0]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label = le.fit_transform(training_labels)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris.data, iris.target)


import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True, rounded=True,
special_characters=True)
graph = graphviz.Source(dot_data)


graph.view('Iris')

from sklearn.metrics import confusion_matrix

print(confusion_matrix(iris.target, ))

