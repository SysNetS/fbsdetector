#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense

# Load dataframe
df = pd.read_csv('../dataset/fbs_nas.csv')

# Preprocessing
target_column = 'label'
features = df.drop(columns=[target_column])
labels = df[target_column]

# Convert labels to numeric values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
def non_shuffling_train_test_split(X, y, split_at, test_size=0.2):
  i = split_at
  X_train, X_test = np.split(X, [i])
  y_train, y_test = np.split(y, [i])
  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = non_shuffling_train_test_split(features, labels, split_at=526, test_size = 0.33)

# Reshape the input data to 3D for 1D CNN
n_timesteps = X_train.shape[1]
n_features = 1  # Assuming each timestep has a single feature
X_train = X_train.values.reshape((X_train.shape[0], n_timesteps, n_features))
X_test = X_test.values.reshape((X_test.shape[0], n_timesteps, n_features))

# Define the deep learning model with 1D CNN layers
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

