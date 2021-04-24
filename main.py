# Libraries
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from PIL import Image
import pandas as pd
import numpy as np

# Load train data
train = pd.read_csv('./data/mnist_train.csv')

# Load test data
test = pd.read_csv('./data/mnist_test.csv')

# Splitting data, train X and train y
X_train = train.drop(labels=["label"] ,axis=1)
y_train = train["label"]

# Splitting data, test X and test y
X_test = test.drop(labels=["label"] ,axis=1)
y_test = test["label"]

# Reshape the array(4d -> 2d)
X_train = X_train.values.reshape(-1,28*28)
X_test = X_test.values.reshape(-1,28*28)

X_train = X_train/256
X_test = X_test/256

# Define neural network classification model
clf = MLPClassifier(solver='adam', activation='relu',hidden_layer_sizes=(64,64))

# Train the model
clf.fit(X_train, y_train)

# Predictions
predictions = clf.predict(X_test)

# Evaluation
accuracy = confusion_matrix(y_test, predictions)

def getAccuracy(cm):
  diagonal = cm.trace()
  el = cm.sum()
  return diagonal/el

print("Accuracy: ", getAccuracy(accuracy))
print("=" * 40)
# ===================================================================

# Open image file
img = Image.open('./images/Six.png')

# Get image data
img_data = list(img.getdata())

for i in range(len(img_data)):
  img_data[i] = 255 - img_data[i]

num = np.array(img_data)/256

p = clf.predict([num])
print(p)
