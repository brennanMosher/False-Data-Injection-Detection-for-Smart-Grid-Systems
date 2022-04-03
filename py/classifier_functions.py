import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MinMaxScaler



def read_dataset(loc):
	'''
	:param loc: Dataset location in directory
	:return: Pandas dataframe containing full dataset
	'''
	return pd.read_csv(loc)


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


def training(train_df, label_train_df, classifier):
	'''
	Runs training algorithm on the dataset

	:param train_df: Full training set
	:param label_train_df: Labels for the training set
	:param classifier: Specifies the classifier used for training
	:return: Trained classifier
	'''


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
	print(confusion_matrix(test_labels, model_prediction))
