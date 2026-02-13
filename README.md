# Human & Animal Detection + OCR System

A complete AI system for detecting and classifying humans/animals in videos, plus extracting text from industrial stenciled images. Built with PyTorch, PaddleOCR, and Streamlit.

---

## Assignment Overview

This project addresses real-world AI system design under offline and industrial constraints:

**Part A: Human & Animal Detection**
- Two-stage pipeline: Object Detection → Classification
- Processes videos automatically from `./test_videos/`
- Outputs annotated videos to `./outputs/`

**Part B: Offline OCR for Stenciled/Industrial Text**
- Extracts text from faded/painted industrial markings
- Works 100% offline (no cloud APIs)
- Handles low contrast and surface damage

---

## Project Structure

```
project/
├── train_detector.py              # Detector training script
├── train_classifier.py            # Classifier training script
├── inference_detector.py          # Batch video processing
├── paddle_ocr_crate_fixed.py     # Command-line OCR tool
├── streamlit_app.py              # Web interface
├── requirements.txt              # Dependencies
│
├── datasets/                     # Training data
│   ├── detector/
│   │   └── train/
│   │       ├── images/
│   │       └── _annotations.coco.json
│   └── classifier/
│       ├── train/ (human/, animal/)
│       └── val/ (human/, animal/)
│
├── models/                       # Trained weights
│   ├── detector/
│   │   └── fasterrcnn_4gb_epoch_10.pth
│   └── classifier/
│       └── best_classifier.pth
│
├── test_videos/                  # Input videos
└── outputs/                      # Processed results
    ├── detector/
    └── classifier/
```

---

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Selection & Justification

### Part A: Detection Dataset

**Source:** Roboflow Universe (Custom annotated dataset)  
**Format:** COCO JSON  
**Classes:** Human, Animal

**Justification:**
- **Not COCO/ImageNet** (meets assignment requirement)
- **High-quality annotations** - Professional bounding boxes
- **Domain-specific** - Focused on surveillance/wildlife scenarios
- **Balanced classes** - Equal representation prevents bias
- **Annotation quality** - Clean, verified labels

**Why NOT COCO/ImageNet:**
- Generic datasets lack human-animal granularity
- Too many unnecessary classes (80+ in COCO)
- Custom dataset provides better control
- Improved performance for specific use case

---

### Part A: Classifier Dataset

**Format:** ImageFolder (PyTorch standard)
```
classifier/
├── train/
│   ├── human/  (~3000 images)
│   └── animal/ (~3000 images)
└── val/
    ├── human/  (~500 images)
    └── animal/ (~500 images)
```

**Justification:**
- Simple structure, easy to maintain
- Binary classification perfect for task
- Balanced dataset prevents bias
- Augmentation-friendly

---

## Model Selection & Justification

### Part A: Detector - Faster R-CNN + MobileNetV3-Large-FPN

**Why Faster R-CNN?**
1. **Two-stage detection** → More accurate than one-stage
2. **Region proposals** → Better localization for overlapping objects
3. **Industry-standard** → Proven architecture
4. **NOT YOLO** → Meets assignment constraint

**Why MobileNetV3-Large Backbone?**
1. **Efficient** - Fast inference while maintaining accuracy
2. **Mobile-friendly** - Deployable on edge devices
3. **FPN** - Handles multi-scale objects
4. **Pre-trained** - Faster convergence with transfer learning
5. **Memory efficient** - Lower GPU requirements

**Training Strategy:**
```python
# Freeze backbone for efficient training
for param in model.backbone.parameters():
    param.requires_grad = False

# Only train detection head
# Mixed precision (AMP) for memory efficiency
# Batch size = 1 (GPU memory constraint)
# SGD optimizer for better generalization
```

---

### Part A: Classifier - EfficientNet-B0

**Why EfficientNet-B0?**
1. **State-of-the-art efficiency** - Best accuracy per FLOP
2. **Small model** - ~5.3M parameters, fast inference
3. **Transfer learning** - Pre-trained on ImageNet
4. **Mobile-ready** - Resource-constrained friendly
5. **Proven performance** - Excellent for binary classification

**Two-Phase Training Strategy:**

**Phase 1: Freeze Backbone (3 epochs)**
```python
for param in model.features.parameters():
    param.requires_grad = False
optimizer = Adam(lr=1e-4)
```
- Fast initial convergence
- Prevents catastrophic forgetting
- Head learns task-specific features

**Phase 2: Fine-tune All (7 epochs)**
```python
for param in model.parameters():
    param.requires_grad = True
optimizer = Adam(lr=1e-5)  # Lower LR
```
- Adapts backbone to domain
- Higher final accuracy
- Prevents overfitting

---

### Part B: OCR - PaddleOCR

**Why PaddleOCR?**
1. **100% Offline** - Works without internet after setup
2. **Degraded text** - Excellent for faded/stenciled text
3. **Rotation invariant** - Handles rotated text
4. **Open source** - Free, active community
5. **Industrial-grade** - Production-proven

**Components:**
- **DB Model** - Text detection
- **CRNN** - Character recognition
- **Orientation Classifier** - Handles rotation

---

## Training Pipeline

### Detector Training

```python
# 1. Load COCO dataset
train_dataset = CustomCocoDataset(
    root="datasets/detector/train/images",
    annFile="datasets/detector/train/_annotations.coco.json"
)

# 2. Initialize model
model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

# 3. Freeze backbone
for p in model.backbone.parameters():
    p.requires_grad = False

# 4. Replace classifier head
num_classes = 4  # background + human + animal + other
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 5. Training loop with mixed precision
scaler = torch.cuda.amp.GradScaler()
for epoch in range(10):
    for images, targets in train_loader:
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    wandb.log({"epoch": epoch, "train_loss": loss})
    torch.save(model.state_dict(), f"fasterrcnn_epoch_{epoch}.pth")
```

**Logged Metrics:**
- Training loss per epoch
- Classification loss
- Bounding box regression loss
- RPN losses

---

### Classifier Training

```python
# Phase 1: Train head only (3 epochs)
model = efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(in_features, 2)

for param in model.features.parameters():
    param.requires_grad = False

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Train for 3 epochs...

# Phase 2: Fine-tune all (7 epochs)
for param in model.parameters():
    param.requires_grad = True

optimizer = Adam(model.parameters(), lr=1e-5)

# Train for 7 epochs...
# Save best model based on validation accuracy
```

**Logged Metrics:**
- Training/validation loss
- Training/validation accuracy  
- Phase indicator (freeze/fine-tune)
- Best validation accuracy

---

## Inference Pipeline

### Automatic Video Processing

```bash
# Place videos in test_videos/
python inference_detector.py

# Output structure:
outputs/
├── detector/
│   └── video.mp4    # Yellow bounding boxes
└── classifier/
    └── video.mp4    # Green (Human) / Red (Animal) labels
```

**Pipeline Flow:**
```
INPUT: test_videos/*.mp4
    ↓
Load Models (Detector + Classifier)
    ↓
For each frame:
    ├─→ Detector: Find bounding boxes
    │   └─→ Save with yellow boxes
    │
    └─→ For each box:
        ├─→ Crop region
        ├─→ Classifier: Predict class
        └─→ Draw colored box + label
            └─→ Green = Human
            └─→ Red = Animal
    ↓
OUTPUT: Two videos per input
```

**Confidence Threshold:** 0.5 (configurable)

---

### OCR Processing

```bash
python paddle_ocr_crate_fixed.py image.jpg

# Output:
# - Extracted text (console + clipboard)
# - Visualization with bounding boxes (ocr_result.jpg)
```

**Preprocessing Pipeline:**
```python
# 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
# → Enhances contrast in low-light regions
# → Adaptive per region
# → Prevents noise amplification

# 2. Sharpening Filter
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv2.filter2D(enhanced, -1, kernel)
# → Enhances edges and text boundaries
# → Makes faded text readable
```

**Why This Preprocessing?**
- Industrial text often has low contrast
- Faded paint requires enhancement
- Stenciled text has sharp edges

---

## Results

### Detection Performance
- Training Loss (Final): 0.2156
- Inference Speed (GPU): ~15 FPS
- Inference Speed (CPU): ~2 FPS

### Classification Performance
| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 90.22% | 90.08% |
| Loss | 0.31 | 0.32 |

### OCR Performance
- Clean text: 95%+ success
- Faded text: 80-85% success
- Rotated text: 90%+ success

---

## Challenges & Solutions

### Challenge 1: GPU Memory Constraints
**Problem:** CUDA out of memory during detector training

**Solutions:**
```python
# 1. Batch size = 1
BATCH_SIZE = 1

# 2. Mixed precision (AMP)
scaler = torch.cuda.amp.GradScaler()

# 3. Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False
```

---

### Challenge 2: Empty Bounding Boxes
**Problem:** Some images have invalid boxes (w=0, h=0)

**Solution:**
```python
def __getitem__(self, idx):
    while True:
        img, target = super().__getitem__(idx)
        boxes = []
        for obj in target:
            x, y, w, h = obj["bbox"]
            if w > 1 and h > 1:  # Filter invalid
                boxes.append([x, y, x+w, y+h])
        
        if len(boxes) == 0:
            idx = (idx + 1) % len(self)  # Try next
            continue
        return img, {"boxes": torch.tensor(boxes), ...}
```

---

### Challenge 3: Video Codec Compatibility
**Problem:** Videos don't play in browsers

**Solution:**
```python
# Read as bytes for Streamlit
with open(video_path, 'rb') as f:
    video_bytes = f.read()
st.video(video_bytes)

# Optional: Convert with ffmpeg
subprocess.run(['ffmpeg', '-i', 'input.mp4', 
                '-vcodec', 'libx264', 'output.mp4'])
```

---

### Challenge 4: Faded OCR Text
**Problem:** PaddleOCR fails on low-contrast text

**Solution:**
```python
# Enhanced preprocessing
enhanced = enhance_image(img)  # CLAHE + Sharpening
result = ocr.ocr(enhanced, cls=True)
# Result: Confidence improved from 0.3 → 0.85+
```

---

## Possible Improvements

### Model Improvements
1. **Ensemble Detection** - Combine multiple detectors
2. **Attention Mechanisms** - Add to classifier
3. **Better Augmentation** - Use Albumentations

### Pipeline Improvements
1. **Multi-GPU Training** - Parallel processing
2. **Real-time Processing** - Skip frames for speed
3. **Model Quantization** - Faster inference

### OCR Improvements
1. **Custom Training** - Fine-tune on industrial text
2. **Multi-scale Processing** - Handle various sizes
3. **Language Model** - Post-processing spell-check

---

## Streamlit Web Interface

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```



## Requirements:
- requirements.txt

---

## Troubleshooting

### Models Not Loading
```bash
# Check paths
ls models/detector/fasterrcnn_4gb_epoch_10.pth
ls models/classifier/best_classifier.pth
```

### CUDA Out of Memory
```python
# Use CPU
DEVICE = torch.device("cpu")
```

### Videos Not Playing
```bash
# Try different browser (Chrome recommended)
# Or just download and play locally
```


