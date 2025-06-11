import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_label_studio_json(filename):
    # Load your Label Studio export
    with open(filename, 'r') as f:
        data = json.load(f)

    # Collect annotation intervals
    rows = []

    for task in data:
        for ann in task.get('annotations', []):
            for result in ann.get('result', []):
                if result.get('type') == 'timeserieslabels':
                    value = result['value']
                    start = value['start']
                    end = value['end']
                    label = value['timeserieslabels'][0]

                    # Normalize label
                    label_normalized = label.upper().replace("TREND", "")

                    rows.append({
                        'Start': start,
                        'End': end,
                        'Type': label_normalized
                    })

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Convert 'Start' column to datetime
    df['Start'] = pd.to_datetime(df['Start'])

    # Optional: Convert 'End' column too
    df['End'] = pd.to_datetime(df['End'])

    # Sort by 'Start'
    df = df.sort_values(by='Start').reset_index(drop=True)
    return df


def make_labeled_data(df, labels_df):
    labels_df['Start'] = pd.to_datetime(labels_df['Start'])
    labels_df['End'] = pd.to_datetime(labels_df['End'])
    df['UpTrend'] = np.nan
    df['DownTrend'] = np.nan
    for idx, row in labels_df.iterrows():
        start_date = row['Start']
        end_date = row['End']
        trend_type = row['Type'].strip().upper()
        label = f"{trend_type}"
        if trend_type == 'UP':
            df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), 'UpTrend'] = label
            df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), 'DownTrend'] = np.nan
        elif trend_type == 'DOWN':
            df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), 'DownTrend'] = label
            df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date), 'UpTrend'] = np.nan
    return df


# Sort by date to ensure chronological order
def get_window_dataset(df, window_size):
    df = df.sort_values('Date').reset_index(drop=True)

    # Ensure 'Up Trend' and 'Down Trend' columns are strings
    df['UpTrend'] = df['UpTrend'].astype(str)
    df['DownTrend'] = df['DownTrend'].astype(str)

    windows = []
    labels = []

    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size]
        windows.append(window[['Close']].values)
        
        up_count = (window['UpTrend'] == 'UP').sum()
        down_count = (window['DownTrend'] == 'DOWN').sum()
        if up_count > (window_size // 2):
            labels.append('UP')
        elif down_count > (window_size // 2):
            labels.append('DOWN')
        else:
            labels.append('MIXED')
    return windows, labels