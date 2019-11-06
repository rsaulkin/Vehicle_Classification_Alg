# Check predictions of the different classification algs

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os

# Extract Color Stat method
def extractColorStats(image):
	# split the input image into its respective RGB color channels
	# and then create a feature vector with 6 values: the mean and
	# standard deviation for each of the 3 channels, respectively
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]

	# return our set of features
	return features


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="DS",
	help="path to directory containing the vehicles dataset")
ap.add_argument("-m", "--model", type=str, default="knn",
	help="type of python machine learning model to use")
args = vars(ap.parse_args())

# Define the dictionary of models our script can use, where the key
# to the dictionary is the name of the model (supplied via command
# line argument) and the value is the model itself
models = {
	"knn": KNeighborsClassifier(n_neighbors=1),
	"adaBoost": AdaBoostClassifier(),
	"svm": SVC(kernel="linear"),
	"decision_tree": DecisionTreeClassifier(),
	"random_forest": RandomForestClassifier(n_estimators=100),
	"neural_network": MLPClassifier()
}

# Grab all image paths in the input dataset directory, initialize our
# list of extracted features and corresponding labels
print("Extracting image features...")
imagePaths = paths.list_images(args["dataset"])
imgData = []
classLabels = []

# Loop over our input images
for imagePath in imagePaths:
	# load the input image from disk, compute color channel
	# statistics, and then update our data list
	image = Image.open(imagePath)
	features = extractColorStats(image)
	imgData.append(features)

	# Extract the class label from the file path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	classLabels.append(label)

# Encode the labels, converting them from strings to integers
encodedClassLabels = LabelEncoder()
labels = encodedClassLabels.fit_transform(classLabels)

# Perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(imgData, classLabels, test_size=0.25)

# train the model
print("Using '{}' model".format(args["model"]))
print("========================")
model = models[args["model"]]
model.fit(trainX, trainY)

# Make predictions on our data and show a classification report
print("Perform predictions...")
predictions = model.predict(testX)

# Print the prediction report
print(classification_report(testY, predictions, target_names=encodedClassLabels.classes_))