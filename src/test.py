
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


from src.preprocess.make_dataset import get_window_dataset
from src.preprocess.feature import normalize_windows

# Load the data
df = pd.read_csv('data/data.csv', parse_dates=['Date'])[:200]
windows, labels = get_window_dataset(df)

X = np.array(windows)
X = normalize_windows(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = to_categorical(y, num_classes=3)

# Load
model = tf.keras.models.load_model('models/cnn_1d_500.keras')


# Step 1: Predict class probabilities
y_pred_probs = model.predict(X)

# Step 2: Convert predictions and true labels to class indices
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y, axis=1)

# Step 3: Generate the report
original_class_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
print(classification_report(y_true, y_pred, target_names=original_class_names))
