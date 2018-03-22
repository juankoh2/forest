from sklearn import tree

import pandas as pd
mydata= pd.read_csv('training.csv')
training_labels = mydata.iloc[1:198,2]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label = le.fit_transform(training_labels)

clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(mydata, mydata.type)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=mydata.columns,
class_names=["Sugi","Hinoki","mixed deciduous","non-forest land"])
graph = graphviz.Source(dot_data)


graph.view('Forest')



print(mydata.type)



import os

os.system("PAUSE")
