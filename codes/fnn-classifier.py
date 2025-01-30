#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataframe
df = pd.read_csv('../dataset/fbs_nas.csv')
# df = pd.read_csv('../dataset/fbs_nas_ext.csv')
# df = pd.read_csv('../dataset/fbs_rrc.csv')
# df = pd.read_csv('../dataset/msa_nas.csv')
# df = pd.read_csv('../dataset/msa_nas_ext.csv')
# df = pd.read_csv('../dataset/msa_rrc.csv')

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

# Define the deep learning model (FNN)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
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

