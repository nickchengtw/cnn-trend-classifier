import argparse

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')  # Or 'QtAgg' depending on your system

from src.preprocess.feature import *
from src.preprocess.make_dataset import *
from src.utils.plot import plot_labeled


# Argument parser
parser = argparse.ArgumentParser(description="Trend prediction with CNN")
parser.add_argument('-c', '--csv_path', type=str, default='data/data.csv', help='Path to input CSV file')
parser.add_argument('-m', '--model_path', type=str, default='models/cnn_1d_huge.keras', help='Path to trained model')
parser.add_argument('-l', '--limit', type=int, default=180, help='Number of rows to load from CSV')

args = parser.parse_args()

# Constants
WINDOW_SIZE = 15
MIXED_LABEL = 1

# Load the data
df_det = pd.read_csv(
    args.csv_path,
    parse_dates=['Date'],
    index_col=0,
    dtype={'Up Trend': 'object', 'Down Trend': 'object'}
)[:args.limit]

windows, labels = get_window_dataset(df_det)

X = np.array(windows)
X = normalize_windows(X)

# Load model
model = tf.keras.models.load_model(args.model_path)
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