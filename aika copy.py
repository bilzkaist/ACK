import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Load the data from the csv file
data = pd.read_csv("indoor_outdoor_data.csv")

# Extract the features and labels from the data
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

# Define the neural network model
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Use the trained model to make predictions on new data
predictions = model.predict(X_test)

# Convert the predictions to binary values (1 for indoor, 0 for outdoor)
predictions = np.round(predictions).flatten()

# Calculate the accuracy of the model compared to the ground truth
accuracy = np.sum(predictions == y_test) / len(y_test)
print("Accuracy Compared to Ground Truth:", accuracy)
