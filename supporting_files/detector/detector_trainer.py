import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import wandb

# W&B INIT
wandb.init(
    project="human-animal-detection",
    name="fasterrcnn-4gb-safe",
    config={"epochs": 10, "batch_size": 1, "lr": 0.002}
)

cfg = wandb.config

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# DATASET
class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile)
        self.transform = transform
        cat_ids = self.coco.getCatIds()
        self.cat2idx = {c: i+1 for i, c in enumerate(cat_ids)}

    def __getitem__(self, idx):
        while True:   # keep trying until non-empty sample
            img, target = super().__getitem__(idx)

            boxes, labels = [], []
            for obj in target:
                x, y, w, h = obj["bbox"]
                if w > 1 and h > 1:  # skip invalid boxes
                    boxes.append([x, y, x+w, y+h])
                    labels.append(self.cat2idx[obj["category_id"]])

            if len(boxes) == 0:
                idx = (idx + 1) % len(self)
                continue  # retry next image

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels}

            img = F.resize(img, (480, 480))
            img = F.to_tensor(img)

            return img, target


# DATA
train_dataset = CustomCocoDataset(
    root=r"datasets\detector\train\images",
    annFile=r"datasets\detector\train\_annotations.coco.json"
)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

# MODEL
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

for p in model.backbone.parameters():
    p.requires_grad = False

in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = len(train_dataset.cat2idx) + 1
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

# OPTIMIZER
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9)
scaler = torch.cuda.amp.GradScaler()

# TRAIN LOOP
for epoch in range(cfg.epochs):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    wandb.log({"epoch": epoch+1, "train_loss": total_loss})
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), f"fasterrcnn_4gb_epoch_{epoch+1}.pth")

wandb.finish()
print("DONE 🚀")
