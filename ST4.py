import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
train_data = pd.read_csv('train4.csv')
test_data = pd.read_csv('test4.csv')

# Split the data into features and labels
X_train = train_data[['x', 'y']].values
y_train_color = train_data['color'].values
y_train_marker = train_data['marker'].values

X_test = test_data[['x', 'y']].values
y_test_color = test_data['color'].values
y_test_marker = test_data['marker'].values

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform label encoding for color and marker labels
label_encoder_color = LabelEncoder()
label_encoder_marker = LabelEncoder()
y_train_color_encoded = label_encoder_color.fit_transform(y_train_color)
y_train_marker_encoded = label_encoder_marker.fit_transform(y_train_marker)
y_test_color_encoded = label_encoder_color.transform(y_test_color)
y_test_marker_encoded = label_encoder_marker.transform(y_test_marker)

# Build the neural network
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(label_encoder_color.classes_), activation='softmax') # Output layer for color prediction
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for color prediction
model.fit(X_train, y_train_color_encoded, epochs=100, batch_size=32)

# Evaluate the model on the test data for color prediction
test_loss, test_acc_color = model.evaluate(X_test, y_test_color_encoded)

# Build another neural network for marker prediction
model_marker = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(label_encoder_marker.classes_), activation='softmax') # Output layer for marker prediction
])

model_marker.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for marker prediction
model_marker.fit(X_train, y_train_marker_encoded, epochs=100, batch_size=32)

# Evaluate the model on the test data for marker prediction
test_loss, test_acc_marker = model_marker.evaluate(X_test, y_test_marker_encoded)

print('Color Test accuracy:', test_acc_color)
print('Marker Test accuracy:', test_acc_marker)
