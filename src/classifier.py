import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from src.preprocess.feature import *
from src.preprocess.make_dataset import *
from src.utils.plot import plot_labeled

import matplotlib
matplotlib.use('TkAgg')  # Or 'QtAgg' depending on your system

# Load the data
df_det = pd.read_csv('data/data.csv', parse_dates=['Date'], index_col=0, dtype={'Up Trend': 'object', 'Down Trend': 'object'})[:180]
windows, labels = get_window_dataset(df_det)

WINDOW_SIZE = 15
MIXED_LABEL = 1

X = np.array(windows)
X = normalize_windows(X)

# Step 1: Predict class probabilities
model = tf.keras.models.load_model('models/cnn_1d_huge.keras')
y_pred_probs = model.predict(X)
# Step 2: Convert predictions and true labels to class indices
y_pred = np.argmax(y_pred_probs, axis=1)

y_pred_padded = np.pad(y_pred, (0, WINDOW_SIZE - 1), mode='constant', constant_values=MIXED_LABEL)


DOWNTREND_LABEL = 0
UPTREND_LABEL = 2

df_det['DownTrend'] = np.where(y_pred_padded == DOWNTREND_LABEL, 'DOWN', '')
df_det['UpTrend'] = np.where(y_pred_padded == UPTREND_LABEL, 'UP', '')


# Step 1: Create masks
mask_down = df_det['DownTrend'] == 'DOWN'
mask_up = df_det['UpTrend'] == 'UP'

# Step 2: Create trend block IDs
df_det['DownTrendID'] = np.nan
df_det.loc[mask_down, 'DownTrendID'] = (
    (~mask_down.shift(fill_value=False)).cumsum()[mask_down]
)

df_det['UpTrendID'] = np.nan
df_det.loc[mask_up, 'UpTrendID'] = (
    (~mask_up.shift(fill_value=False)).cumsum()[mask_up]
)

# Step 3: Assign trend with ID to respective columns
df_det['DownTrend'] = np.where(
    mask_down, 'DOWN_' + df_det['DownTrendID'].astype('Int64').astype(str), ''
)

df_det['UpTrend'] = np.where(
    mask_up, 'UP_' + df_det['UpTrendID'].astype('Int64').astype(str), ''
)

# Step 4: Drop helper columns
df_det.drop(columns=['DownTrendID', 'UpTrendID'], inplace=True)

df_det['UpTrend'].replace('', np.nan, inplace=True)
df_det['DownTrend'].replace('', np.nan, inplace=True)


plot_labeled(df_det.copy())