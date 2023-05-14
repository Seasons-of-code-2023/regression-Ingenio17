import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the data
train_data = pd.read_csv('train3.csv')
test_data = pd.read_csv('test3.csv')

# Split the data into features and labels
X_train = train_data[['x', 'y']].values
y_train = train_data['color'].values

X_test = test_data[['x', 'y']].values
y_test = test_data['color'].values

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform label encoding for color labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Build the neural network
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=100, batch_size=32)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)