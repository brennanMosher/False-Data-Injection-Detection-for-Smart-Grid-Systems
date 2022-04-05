from py import classifier_functions
from py.classifier_functions import *

DATA_LOC = "../../dataset/Original/data_injection_and_normal_events_dataset.csv"

df = read_dataset(DATA_LOC)
print(df.head())

df = dataset_preprocess(df)
print(df.head())

train_df, test_df = training_testing_split(df, testing_split=0.3)

print(train_df.head())

print(test_df.head())

train_df, label_train_df = data_label_split(train_df)

print(train_df.head())

print(label_train_df.head())

test_df, label_test_df = data_label_split(test_df)
print(test_df.head())


print(label_test_df.head())

train_df, test_df, fs = select_features(train_df, label_train_df, test_df, f_classif, 50)
train_df = pd.DataFrame(train_df)
test_df = pd.DataFrame(test_df)
print(train_df.head())
print(test_df.head())

rf = training(train_df, label_train_df, classifier="Random Forest")
testing(test_df, label_test_df, rf)


lr = training(train_df, label_train_df, classifier="Logistic Regression")
testing(test_df, label_test_df, lr)


nb = training(train_df, label_train_df, classifier="Naive Bayes")
testing(test_df, label_test_df, nb)


knn = training(train_df, label_train_df, classifier="KNN")
testing(test_df, label_test_df, knn)


dt = training(train_df, label_train_df, classifier="Decision Tree")
testing(test_df, label_test_df, dt)


ada = training(train_df, label_train_df, classifier="AdaBoost")
testing(test_df, label_test_df, ada)


gb = training(train_df, label_train_df, classifier="Gradient Boosting")
testing(test_df, label_test_df, gb)


bag = training(train_df, label_train_df, classifier="Bagging")
testing(test_df, label_test_df, bag)


svm = training(train_df, label_train_df, classifier="SVM")
testing(test_df, label_test_df, svm)



