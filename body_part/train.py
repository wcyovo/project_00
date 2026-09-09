import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import BodyPartDataset, JSON_PATH
from model import BodyPartRegressor

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IOU_THRESHOLD = 0.5
EARLY_STOP_TARGET = 0.95

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

def calculate_accuracy(pred, target, iou_threshold=IOU_THRESHOLD):
    batch_size = pred.size(0)
    correct = 0
    for i in range(batch_size):
        avg_iou = 0
        for j in range(0, 20, 4):
            avg_iou += compute_iou(pred[i, j:j+4], target[i, j:j+4])
        avg_iou /= 5
        if avg_iou > iou_threshold:
            correct += 1
    return correct / batch_size

train_dataset = BodyPartDataset(JSON_PATH, train=True)
val_dataset = BodyPartDataset(JSON_PATH, train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = BodyPartRegressor().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.SmoothL1Loss()

best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        pressure = batch['pressure'].to(DEVICE)
        target = batch['target'].to(DEVICE)
        sleep_pose = batch['sleep_pose'].squeeze().to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(pressure, sleep_pose)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    model.eval()
    val_acc = 0
    with torch.no_grad():
        for batch in val_loader:
            pressure = batch['pressure'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            sleep_pose = batch['sleep_pose'].squeeze().to(DEVICE)
            pred = model(pressure, sleep_pose)
            val_acc += calculate_accuracy(pred, target)
    val_acc /= len(val_loader)
    
    scheduler.step(val_acc)
    
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    if val_acc >= EARLY_STOP_TARGET:
        torch.save(model.state_dict(), 'best_model_95.pth')
        break

print(f"最佳验证准确率: {best_acc:.4f}")