import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

UP_TREND_COL = 'UpTrend'
DOWN_TREND_COL = 'DownTrend'


def plot_closing_price(df, start_date, end_date):
    """
    Plot closing price using Seaborn between two dates from a pre-loaded DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'Date' and 'Close' columns
    - start_date (str or datetime): Start date (e.g., '2016-01-01')
    - end_date (str or datetime): End date (e.g., '2016-06-01')
    """
    # Filter data
    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                     (df['Date'] <= pd.to_datetime(end_date))]

    # Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_filtered, x='Date', y='Close')
    plt.title(f'Closing Price from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_labeled(df):
    df.reset_index(inplace=True)

    plt.figure(figsize=(20, 10))

    ax = sns.lineplot(x=df.index, y=df['Close'])
    ax.set(xlabel='Date')

    labels = df[UP_TREND_COL].dropna().unique().tolist()

    for label in labels:
        sns.lineplot(x=df[df[UP_TREND_COL] == label].index,
                    y=df[df[UP_TREND_COL] == label]['Close'],
                    color='green')

        ax.axvspan(df[df[UP_TREND_COL] == label].index[0],
                df[df[UP_TREND_COL] == label].index[-1],
                alpha=0.2,
                color='green')

    labels = df[DOWN_TREND_COL].dropna().unique().tolist()

    for label in labels:
        sns.lineplot(x=df[df[DOWN_TREND_COL] == label].index,
                    y=df[df[DOWN_TREND_COL] == label]['Close'],
                    color='red')

        ax.axvspan(df[df[DOWN_TREND_COL] == label].index[0],
                df[df[DOWN_TREND_COL] == label].index[-1],
                alpha=0.2,
                color='red')
                
    locs, _ = plt.xticks()
    labels = []

    for position in locs[1:-1]:
        idx = int(round(position))
        if 0 <= idx < len(df):
            labels.append(str(df['Date'].iloc[idx])[:-9])
        else:
            labels.append('')

    plt.xticks(locs[1:-1], labels)
    plt.show()


def plot_training(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()