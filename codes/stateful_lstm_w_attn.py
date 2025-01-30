import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Attention, Concatenate, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(csv_file, label_column):
    data = pd.read_csv(csv_file)
    features = data.drop(columns=[label_column])
    labels = data[label_column]
    return features, labels

def preprocess_data(features, labels, seq_length):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if labels.dtype == 'object':
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)

    num_samples = len(features_scaled) // seq_length
    features_reshaped = features_scaled[:num_samples * seq_length].reshape(num_samples, seq_length, -1)
    labels_reshaped = labels[:num_samples * seq_length].reshape(num_samples, seq_length)

    labels_final = labels_reshaped[:, -1]

    return features_reshaped, labels_final

class StatefulLSTM:
    def __init__(self, units, seq_length):
        self.units = units
        self.seq_length = seq_length
        self.lstm = LSTM(units, stateful=True, return_sequences=True)

    def __call__(self, inputs):
        batch_size = tf.shape(inputs)[0]
        h_t = tf.zeros((batch_size, self.units))
        c_t = tf.zeros((batch_size, self.units))

        outputs = []
        for t in range(self.seq_length):
            x_t = inputs[:, t, :]
            h_t, c_t = self.lstm(x_t[tf.newaxis, :, :], initial_state=[h_t, c_t])
            outputs.append(h_t)

        return tf.stack(outputs, axis=1)

class LSTMwithAttention:
    def __init__(self, units):
        self.units = units
        self.lstm = LSTM(units, return_sequences=True)
        self.attention = Attention()
        self.dense = Dense(units, activation='tanh')

    def __call__(self, inputs):
        H = self.lstm(inputs)
        context_vector = self.attention([H, H])
        h_t = H[:, -1, :]
        h_t_prime = self.dense(Concatenate()([context_vector, h_t]))
        return h_t_prime

def create_model(input_shape, lstm_units, seq_length):
    inputs = Input(shape=input_shape)
    stateful_lstm = StatefulLSTM(lstm_units, seq_length)
    h_t = stateful_lstm(inputs)
    lstm_attention = LSTMwithAttention(lstm_units)
    h_t_prime = lstm_attention(h_t)
    outputs = Dense(1, activation='sigmoid')(h_t_prime)  # Binary classification
    model = Model(inputs, outputs)
    return model

def main(csv_file, label_column, seq_length=10, lstm_units=128, test_size=0.2, random_state=42):
    # Load data
    features, labels = load_data(csv_file, label_column)

    # Preprocess data
    features_reshaped, labels_final = preprocess_data(features, labels, seq_length)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_reshaped, labels_final, test_size=test_size, random_state=random_state)

    # Create the model
    input_shape = (seq_length, X_train.shape[2])
    model = create_model(input_shape, lstm_units, seq_length)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main("../dataset/fbs_nas.csv", "label")