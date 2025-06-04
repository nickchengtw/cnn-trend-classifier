def augment_data(windows, labels):
    augmented_windows = []
    augmented_labels = []

    for i in range(len(windows)):
        original_window = windows[i]
        label = labels[i]
        
        # Flip prices vertically (around a baseline)
        flipped_window = original_window.copy()
        # Flip OHLC values: we reverse price movement
        # Mirror around the first close price (or another reference)
        base_price = original_window[0][-1]  # first close
        flipped_window = base_price - (original_window - base_price)

        # Reverse the label
        if label == 'UP':
            new_label = 'DOWN'
        elif label == 'DOWN':
            new_label = 'UP'
        else:
            new_label = 'MIXED'

        # Save augmented data
        augmented_windows.append(flipped_window)
        augmented_labels.append(new_label)

    windows += augmented_windows
    labels += augmented_labels

    # Augmentation loop (new)
    augmented_windows = []
    augmented_labels = []

    for window, label in zip(windows, labels):
        reversed_window = window[::-1]  # reverse time order
        
        if label == 'UP':
            flipped_label = 'DOWN'
        elif label == 'DOWN':
            flipped_label = 'UP'
        else:
            flipped_label = 'MIXED'
        
        augmented_windows.append(reversed_window)
        augmented_labels.append(flipped_label)

    # Combine both
    windows = windows + augmented_windows
    labels = labels + augmented_labels
    return windows, labels