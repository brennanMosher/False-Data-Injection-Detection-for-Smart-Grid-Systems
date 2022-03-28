from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

#Dataset

# dataset = pd.read_csv(".\dataset\\data_injection_and_normal_events_dataset.csv")
# X = dataset.iloc[:, 0:129].values
# y = dataset.iloc[:, 129].values

training_dataset = pd.read_csv(".\\dataset\\seventy_thirty\\training_data_injection_and_normal_events_dataset.csv")
testing_dataset = pd.read_csv(".\\dataset\\seventy_thirty\\testing_data_injection_and_normal_events_dataset.csv")

##Replace inf values with 0
training_dataset.replace([np.inf, -np.inf], 0, inplace=True)
testing_dataset.replace([np.inf, -np.inf], 0, inplace=True)

testing_dataset.head()
training_dataset.head()

X_train = training_dataset.iloc[:, 0:129].values
X_test = testing_dataset.iloc[:, 0:129].values
y_train = training_dataset.iloc[:, 129].values
y_test = testing_dataset.iloc[:, 129].values

#Features Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Classifiers
##Random Forest
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
