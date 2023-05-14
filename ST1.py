import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Load the dataset
data = pd.read_csv("House_Rent_Dataset.csv")

# Select the relevant features and target variable
features = ['BHK', 'Size', 'Floor', 'Area Type', 'Area Locality', 'City', 'Furnishing Status', 'Tenant Preferred', 'Bathroom', 'Point of Contact']
target = 'Rent'

# Preprocess the data
label_encoder = LabelEncoder()
scaler = StandardScaler()

data["Floor"] = label_encoder.fit_transform(data["Floor"])
data["Area Type"] = label_encoder.fit_transform(data["Area Type"])
data["Area Locality"] = label_encoder.fit_transform(data["Area Locality"])
data["City"] = label_encoder.fit_transform(data["City"])
data["Furnishing Status"] = label_encoder.fit_transform(data["Furnishing Status"])
data["Tenant Preferred"] = label_encoder.fit_transform(data["Tenant Preferred"])
data["Point of Contact"] = label_encoder.fit_transform(data["Point of Contact"])

# Split the data into training and testing sets
X = data[features]
y = data[target]

# Scale the numerical features
X_scaled = scaler.fit_transform(X)

# Split the scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.05))

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
