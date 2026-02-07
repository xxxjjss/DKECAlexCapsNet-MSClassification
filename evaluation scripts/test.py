import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def evaluate_model(model, test_loader, device, num_classes=4):
    all_preds = []
    all_labels = []
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            for i in range(num_classes):
                idx = (target == i)
                class_correct[i] += (preds[idx] == target[idx]).sum().item()
                class_total[i] += idx.sum().item()
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    class_accuracies = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    return accuracy, class_accuracies, precision, recall, f1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlexCapsNet(device, num_classes=4).to(device)
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()
results = []
for snr in range():
    file_config = []

    signals = []
    labels = []
    for path, label in file_config:
        try:
            data = pd.read_csv(path, header=None).values
            signals.append(data)
            labels.extend([label] * len(data))
        except FileNotFoundError:
            signals = []
            break
    if not signals:
        continue
    X_test = np.vstack(signals)
    y_test = np.array(labels)
    X_test_tensor = torch.tensor(X_test).float().unsqueeze(1)
    y_test_tensor = torch.tensor(y_test).long()
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=False)
    accuracy, class_accuracies, precision, recall, f1 = evaluate_model(model, test_loader, device)
    row = [snr, accuracy]
    row.extend(class_accuracies)
    row.extend(precision)
    row.extend(recall)
    row.extend(f1)
    results.append(row)
columns = ['Overall Acc'] + \
          [f'Class{i} Acc' for i in range(4)] + \
          [f'Class{i} Prec' for i in range(4)] + \
          [f'Class{i} Recall' for i in range(4)] + \
          [f'Class{i} F1' for i in range(4)]
results_df = pd.DataFrame(results, columns=columns)