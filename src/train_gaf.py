import random
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from pyts.image import GramianAngularField

from src.preprocess.make_dataset import get_window_dataset
from src.preprocess.feature import normalize_windows
from src.preprocess.augmentation import augment_data
from src.utils.plot import plot_training


seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

parser = argparse.ArgumentParser(description="Trend prediction with CNN")
parser.add_argument('-c', '--csv_path', type=str, help='Path to input CSV file')
parser.add_argument('-m', '--model_path', type=str, help='Path to trained model')
parser.add_argument('-w', '--window_size', type=int, default=15, help='Size of the sliding window for time series data')
parser.add_argument('-e', '--epochs', type=int, default=300, help='Number of epochs for training')
args = parser.parse_args()

WINDOW_SIZE = args.window_size

df = pd.read_csv(args.csv_path, parse_dates=['Date'])
windows, labels = get_window_dataset(df, WINDOW_SIZE)
windows, labels = augment_data(windows, labels)

X = np.array(windows)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = to_categorical(y, num_classes=3)

X = normalize_windows(X)

X_raw = X.squeeze()
gaf = GramianAngularField(method='summation')
X_gaf = gaf.fit_transform(X_raw)
X_gaf = X_gaf[..., np.newaxis]


X_train, X_test, y_train, y_test = train_test_split(
    X_gaf, y, test_size=0.2, random_state=42, stratify=y
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential


model = Sequential([
    Conv2D(256, (3, 3), activation='relu', input_shape=(WINDOW_SIZE, WINDOW_SIZE, 1)),
    BatchNormalization(),
    Dropout(0.2),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Conv2D(128, (2, 2), activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    GlobalAveragePooling2D(),
    
    Flatten(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')  # for 3 classes: UP, DOWN, MIXED
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

print(f"Training data count: {len(X_train)}, Test data count: {len(X_test)}")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=args.epochs,
    batch_size=256,
    callbacks=[early_stop]
)

y_pred_probs = model.predict(X_test)

y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

original_class_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
print(classification_report(y_true, y_pred, target_names=original_class_names))

plot_training(history)

model.save(args.model_path)
