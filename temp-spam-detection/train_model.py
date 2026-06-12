
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, BatchNormalization
from imblearn.over_sampling import RandomOverSampler

# Step 1: Unzip the dataset
zip_path = "hmnist_28_28_RGB.csv (2)/hmnist_28_28_RGB.csv.zip"
extract_to = "."
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# Step 2: Load and preprocess data
df = pd.read_csv("hmnist_28_28_RGB.csv")

# Train test split
fractions = np.array([0.8, 0.2])
df = df.sample(frac=1)
train_set, test_set = np.array_split(df, (fractions[:-1].cumsum() * len(df)).astype(int))

y_train = train_set['label']
x_train = train_set.drop(columns=['label'])

y_test = test_set['label']
x_test = test_set.drop(columns=['label'])

# Oversample
oversample = RandomOverSampler()
x_train, y_train = oversample.fit_resample(x_train, y_train)

# Reshape
x_train = np.array(x_train, dtype=np.uint8).reshape(-1, 28, 28, 3)
x_test = np.array(x_test, dtype=np.uint8).reshape(-1, 28, 28, 3)

# Step 3: Build model
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model and save
history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), batch_size=32)
model.save('best_model.h5')
