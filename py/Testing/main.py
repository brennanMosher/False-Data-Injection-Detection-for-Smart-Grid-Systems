from py.classifier_functions import *

DATA_LOC = "../dataset/Original/data_injection_and_normal_events_dataset.csv"

df = read_dataset(DATA_LOC)
df.head()


df = dataset_preprocess(df)
print(df.head())

df = shuffle_data(df)
print(df.head())

train, label = data_label_split(df)

# k_fold_train(train, label, 10, "Random Forest")

# k_fold_train(train, label, 10, "Logistic Regression")

# k_fold_train(train, label, 10, "Naive Bayes")

# k_fold_train(train, label, 10, "KNN")

# k_fold_train(train, label, 10, "Decision Tree")

# k_fold_train(train, label, 10, "AdaBoost")

# k_fold_train(train, label, 10, "Gradient Boosting")

# k_fold_train(train, label, 10, "Bagging")


k_fold_train(train, label, 10, "SVM")