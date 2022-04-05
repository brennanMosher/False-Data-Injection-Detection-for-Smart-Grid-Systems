import time
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def read_dataset(loc):
	'''
	:param loc: Dataset location in directory
	:return: Pandas dataframe containing full dataset
	'''
	return pd.read_csv(loc)


def shuffle_data(df):
	'''
	Shuffle the input data frame
	:param df:
	:return:
	'''
	return shuffle(df)

def dataset_preprocess(df):
	'''
	Can add any other preprocessing here if we need to

	:param df: Dataset
	:return: Preprocessed dataset
	'''
	df.replace([np.inf, -np.inf], 0, inplace=True)
	return df


def training_testing_split(df, testing_split):
	'''
	Split trainign and testing sets
	:param df: Dataset
	:param testing_split: Percentage of dataset specified as testing
	:return: Training dataset, Testing dataset
	'''
	train_df, test_df = train_test_split(df, test_size=testing_split)
	return train_df, test_df


def data_label_split(df):
	'''

	:param df: Dataset
	:return: Dataset with removed labels, Labels for dataset
	'''
	label_df = df['marker']
	df.drop('marker', axis=1, inplace=True)
	return df, label_df


def scaling(df, scaler_algorithm="MinMax"):
	'''

	:param df: Dataset
	:param scaler_algorithm: Default set to MinMax but if we want to change this we can with one function
	:return: Scaled Dataste
	'''
	scaler = MinMaxScaler()
	df_scaled = pd.DataFrame(scaler.fit_transform(df))

	return df_scaled


def split_attack_natural(df):
	'''
	Split into Attack and Natural events if ever needed

	:param df: Dataset
	:return:
	'''
	df_attack = df[df.marker == 1]

	df_natural = df[df.marker == 0]

	return df_attack, df_natural


def select_features(X_train, y_train, X_test, function, num_features):

	'''

	:param X_train: Training data
	:param y_train: Training labels
	:param X_test: Testing set
	:param function: Function for
	:return: Training set feature selected, testing set feature selected, feature selection
	'''
	# configure to select all features
	fs = SelectKBest(score_func=function, k=num_features)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs



def k_fold_train(df, label_df, num_split, classifier):
	'''
	Perform full experiment for K-fold cross validation models. Training and testing are run with accuracy as the only metric

	:param df: Training data
	:param label_df: Labels for training data
	:param num_split: Number of K folds
	:param classifier: Specifies classifier for ML
	:return:
	'''

	print(classifier)
	k_fold = KFold(n_splits=num_split, random_state=None)


	# List to hold accuracy of each fold
	accuracy_list = []

	# Using if/else since apparently anything before Python 3.10 doesn't have switch case
	if classifier == "Random Forest":
		classifier_model = RandomForestClassifier()
	elif classifier == "SVM":
		classifier_model = SVC(kernel='linear')
	elif classifier == "Logistic Regression":
		classifier_model = LogisticRegression()
	elif classifier == "Naive Bayes":
		classifier_model = GaussianNB()
	elif classifier == "KNN":
		classifier_model = KNeighborsClassifier(n_neighbors=7)
	elif classifier == "Decision Tree":
		classifier_model = DecisionTreeClassifier(max_depth=3)
	elif classifier == "AdaBoost":
		classifier_model = AdaBoostClassifier()
	elif classifier == "Gradient Boosting":
		classifier_model = GradientBoostingClassifier()
	elif classifier == "Bagging":
		classifier_model = BaggingClassifier(base_estimator=SVC(),
											 n_estimators=10)
	# Default case just run Random Forest
	else:
		classifier_model = RandomForestClassifier()

	start = time.time()
	# Get split indexes for training/testing
	for train_index, test_index in k_fold.split(df):
		print(train_index)

		# Get training and testing sets
		train, test = df.iloc[train_index, :], df.iloc[test_index, :]

		# Split label data
		label_train, label_test = label_df.iloc[train_index], label_df.iloc[test_index]

		# fit model to data
		classifier_model.fit(train, label_train)

		# Predict on test set
		pred_values = classifier_model.predict(test)

		# Get accuracy from model

		acc = accuracy_score(pred_values, label_test)
		accuracy_list.append(acc)

	avg_acc_score = sum(accuracy_list) / num_split

	end = time.time()
	# Print the accuracy result
	print('accuracy of each fold - {}'.format(accuracy_list))
	print('Avg accuracy : {}'.format(avg_acc_score))
	print('Time: ' + str(end-start))


def training(train_df, label_train_df, classifier):
	'''
	Runs training algorithm on the dataset

	:param train_df: Full training set
	:param label_train_df: Labels for the training set
	:param classifier: Specifies the classifier used for training
	:return: Trained classifier
	'''
	print(classifier)

	# Using if/else since apparently anything before Python 3.10 doesn't have switch case
	if classifier == "Random Forest":
		classifier_model = RandomForestClassifier()
	elif classifier == "SVM":
		classifier_model = SVC(kernel='linear')
	elif classifier == "Logistic Regression":
		classifier_model = LogisticRegression()
	elif classifier == "Naive Bayes":
		classifier_model = GaussianNB()
	elif classifier == "KNN":
		classifier_model = KNeighborsClassifier(n_neighbors=7)
	elif classifier == "Decision Tree":
		classifier_model = DecisionTreeClassifier(max_depth=3)
	elif classifier == "AdaBoost":
		classifier_model = AdaBoostClassifier()
	elif classifier == "Gradient Boosting":
		classifier_model = GradientBoostingClassifier()
	elif classifier == "Bagging":
		classifier_model = BaggingClassifier(base_estimator=SVC(),
											 n_estimators=10)
	# Default case just run Random Forest
	else:
		classifier_model = RandomForestClassifier()

	# Training time
	start = time.time()
	# Fit model to data
	classifier_model.fit(train_df, label_train_df)
	end = time.time()

	# Generates feature importance graph and saves it as pdf
	if type(classifier_model).__name__ == "RandomForestClassifier":
		importances = classifier_model.feature_importances_
		sorted_indices = np.argsort(importances)[::-1]
		# plt.figure(figsize=(20, 3)) 
		plt.bar(range(train_df.shape[1]), importances[sorted_indices], align='center')
		plt.xticks(range(train_df.shape[1]), train_df.columns[sorted_indices], rotation=90, fontsize=2)
		plt.tight_layout()
		plt.savefig('importance1.pdf', dpi=600)
		plt.show()

	print("Training time = " + str(round(end - start, 2)) + "s")

	return classifier_model


def testing(test_df, test_labels, classifier_model):
	'''
	Predicts labels for testing set and evaluates the accuracy/precision/recall


	:param test_df: Full testing dataset
	:param test_labels: True labels for testing set
	:param classifier: Trained classifier model
	:return:
	'''
	# Predict on testing data
	start = time.time()
	model_prediction = classifier_model.predict(test_df)
	end = time.time()

	print("Testing time = " + str(round(end - start, 2)) + "s")

	print(classification_report(test_labels, model_prediction))

	# Calculate TP, FN, FP, TN
	# total positives (0s)
	positives = test_labels.size - np.count_nonzero(test_labels)
	# total negatives (1s)
	negatives = np.count_nonzero(test_labels)

	cm = confusion_matrix(test_labels, model_prediction)
	TP = cm[0][0]  # Detecting a 0 when it is a 0
	FN = cm[0][1]  # Detecting a 1 when it is a 0
	FP = cm[1][0]  # Detecting a 0 when it is a 1
	TN = cm[1][1]  # Detecting a 1 when it is a 1

	TPR = round(TP / positives, 3)
	FNR = round(FN / positives, 3)
	FPR = round(FP / negatives, 3)
	TNR = round(TN / negatives, 3)

	print("Total TPs: " + str(TP), "FNs: " + str(FN), "FPs: " + str(FP), "TNs: " + str(TN))
	print("Rates TP: " + str(TPR), "FN: " + str(FNR), "FP: " + str(FPR), "TN: " + str(TNR))

	# Calculate roc_auc. Skips SVC
	if type(classifier_model).__name__ == "SVC":
		print("roc auc skipped")
	else:
		roc_auc = roc_auc_score(test_labels, classifier_model.predict_proba(test_df)[:, 1])
		print("roc auc score: " + str(round(roc_auc, 3)))
