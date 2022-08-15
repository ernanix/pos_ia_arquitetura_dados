# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df1 = pd.read_csv("Immunotherapy.csv",sep=";")
column="Time"
df1[column] = (df1[column] - df1[column].min()) / (df1[column].max() - df1[column].min())    
column="Area"
df1[column] = (df1[column] - df1[column].min()) / (df1[column].max() - df1[column].min())    
column="induration_diameter"
df1[column] = (df1[column] - df1[column].min()) / (df1[column].max() - df1[column].min())    



#sex = {1:"Man",2:"Woman"}
#type_wart = {1:"Common",2:"Plantar",3:"Both"}
#response_treatment = {1:"Yes",0:"No"}

#cores = {0:'red',1:'green'}
#sns.pairplot(df1,vars=df1.head(),hue='Result_of_Treatment',palette=cores)

data = []
target = []

for v in range(len(df1)):
  data_reg = []
  for v2 in range(len(df1.loc[v])-1):
    data_reg.append(df1.loc[v][v2])
  data.append(data_reg)
  target.append(df1.loc[v][-1])

print(data[1])



X_train, X_test, y_train, y_test = train_test_split(data, target,test_size=0.2, random_state=42)

#print(len(y_train))
#print(len(y_test))




classifier = svm.SVC(kernel="linear").fit(X_train, y_train)

print(classifier.predict(X_test))

disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=['no response', 'response'],
        cmap=plt.cm.Blues
)
disp.ax_.set_title("Confusion Matrix")