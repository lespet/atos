

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import svm


filename = "Stroke.csv"
df = pd.read_csv(filename)

df.shape
df.dtypes

# there were problems converting the BMI column
df['BMI'] = pd.to_numeric(df['BMI'],errors='coerce')
df = df.replace(np.nan, 0, regex=True)
df['BMI'] = df['BMI'].astype(float)

obj_data = df.select_dtypes(include=['object']).copy()
le = preprocessing.LabelEncoder()
categorical = list(df.select_dtypes(include=['object']).columns.values)
for cat in categorical:
    print(cat)
    df[cat] = le.fit_transform(df[cat].astype(str))
    df[cat] = le.fit_transform(df[cat])

df.fillna(0)

df.replace(np.nan,0)
y=df
#input features
X=y.iloc[:,1:11]

#actual output to be considered
Y=y['Stroke']


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.7, random_state = 7)

#Creating a svm Classifier
clf = svm.SVC(gamma='scale')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Accuracy,Precision and recall rate for SVM : ')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

c1=confusion_matrix(y_test, y_pred)


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=9, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred_en = clf_entropy.predict(X_test)
print ('Based on entropy criterion the prediction result in decision tree classifier : ')
print (y_pred_en)
X_train.shape
X_test.shape
print ('Accuracy rate for decision tree classifier ')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
c2=confusion_matrix(y_test, y_pred)