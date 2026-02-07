import numpy as np
import pandas as pd
import torch
def load_signals(csv_path):
    df = pd.read_csv(csv_path, header=None)
    return df.values
csv_paths = ["your_dataset"]
signals = []
for path in csv_paths:
    data = load_signals(path)
    signals.append(data[])
signal_lengths = [s.shape[1] for s in signals]
train_originals, test_originals = [], []
for class_signals in signals:
    train = class_signals[]
    test = class_signals[]
    train_originals.append(train)
    test_originals.append(test)
def process_dataset(originals):
    signals = np.vstack(originals)
    labels = np.hstack([np.full(len(s), idx) for idx, s in enumerate(originals)])
    return signals, labels
X_train, y_train = process_dataset(train_originals)
X_test, y_test = process_dataset(test_originals)
train_mean = np.mean(X_train)
train_std = np.std(X_train)
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std
train_signals_tensor = torch.tensor(X_train).float().unsqueeze(1)  # (6400, 1, L)
test_signals_tensor = torch.tensor(X_test).float().unsqueeze(1)  # (1600, 1, L)
train_labels_tensor = torch.tensor(y_train).long()
test_labels_tensor = torch.tensor(y_test).long()
class SignalDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]
train_loader = DataLoader(SignalDataset(train_signals_tensor, train_labels_tensor),
                          batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(SignalDataset(test_signals_tensor, test_labels_tensor),
                         batch_size=BATCH_SIZE, shuffle=False)