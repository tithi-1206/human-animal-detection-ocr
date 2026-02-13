import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

# Configuration
DATA_DIR = r"datasets/classifier"  
BATCH_SIZE = 32
FREEZE_EPOCHS = 3
FINE_TUNE_EPOCHS = 7
LR_HEAD = 1e-4
LR_FINE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize wandb
wandb.init(project="human-animal-classifier")

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Setup
model = torchvision.models.efficientnet_b0(pretrained=True)

# Replace classifier for 2 classes
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 2)

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# Training Function
def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


def validate(model, loader):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total


# Phase 1: Freeze Backbone
for param in model.features.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_HEAD
)

print("Training classifier head...")

for epoch in range(FREEZE_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc = validate(model, val_loader)

    print(f"[Freeze Phase] Epoch {epoch+1}/{FREEZE_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    wandb.log({
        "phase": "freeze",
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

# Phase 2: Fine-tune Entire Model
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINE)

print("Fine-tuning entire model...")

best_val_acc = 0

for epoch in range(FINE_TUNE_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc = validate(model, val_loader)

    print(f"[Fine-Tune Phase] Epoch {epoch+1}/{FINE_TUNE_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    wandb.log({
        "phase": "fine_tune",
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), r"models\best_classifier.pth")

print("Training complete.")
print("Best Validation Accuracy:", best_val_acc)
