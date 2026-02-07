import torch.optim as optim
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        predicted = torch.argmax(output, dim=1)
        correct += (predicted == target).sum().item()
        total += data.size(0)
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy
def test(model, test_loader, criterion, device, num_classes=4):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    misclassified_samples = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            predicted = torch.argmax(output, dim=1)
            correct += (predicted == target).sum().item()
            total += data.size(0)
            all_labels.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            for i in range(num_classes):
                class_mask = (target == i)
                class_correct[i] += (predicted[class_mask] == target[class_mask]).sum().item()
                class_total[i] += class_mask.sum().item()
            for i in range(len(target)):
                if predicted[i] != target[i]:
                    sample_id = idx * test_loader.batch_size + i
                    misclassified_samples.append((sample_id, target[i].item(), predicted[i].item()))
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    class_accuracies = class_correct / class_total
    precision = precision_score(all_labels, all_preds, average=None, labels=np.arange(num_classes), zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, labels=np.arange(num_classes), zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, labels=np.arange(num_classes), zero_division=0)
    return epoch_loss, epoch_accuracy, class_accuracies, precision, recall, f1, misclassified_samples
def train_and_test(model, train_loader, test_loader, num_epochs, optimizer, criterion, device, num_classes=4):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    onecycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,
        steps_per_epoch=len(train_loader),
        epochs=50,
        pct_start=0.3,
        div_factor=30,
    )
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(
            model, train_loader, optimizer, criterion, device, epoch,
            scheduler=onecycle_scheduler if epoch <= 50 else None
        )
        test_metrics = test(
            model, test_loader, criterion, device, num_classes=num_classes
        )
        test_loss, test_accuracy, class_accuracies, precision, recall, f1, misclassified_samples = test_metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexCapsNet(device, num_classes=4).to(device)
criterion = MarginLoss()
optimizer = optim.AdamW(
    model.parameters(),
    weight_decay=2e-4,
    betas=(0.9, 0.999)
)