import os
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input / Output folders
INPUT_FOLDER = "test_videos"
OUTPUT_FOLDER = "outputs"
DETECTOR_OUTPUT = os.path.join(OUTPUT_FOLDER, "detector")
CLASSIFIER_OUTPUT = os.path.join(OUTPUT_FOLDER, "classifier")

os.makedirs(DETECTOR_OUTPUT, exist_ok=True)
os.makedirs(CLASSIFIER_OUTPUT, exist_ok=True)

# Confidence Threshold
CONF_THRESH = 0.5

# Classifier transforms
clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load Detector
detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
num_classes = 4  # include background
in_features = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
detector.load_state_dict(torch.load(r"models\detector\fasterrcnn_4gb_epoch_10.pth", map_location=DEVICE))
detector.to(DEVICE)
detector.eval()
print("Detector loaded ✅")

# Load Classifier
classifier = torchvision.models.efficientnet_b0(weights=None)
in_features = classifier.classifier[1].in_features
classifier.classifier[1] = torch.nn.Linear(in_features, 2)
classifier.load_state_dict(torch.load(r"models\classifier\best_classifier.pth", map_location=DEVICE))
classifier.to(DEVICE)
classifier.eval()
print("Classifier loaded ✅")

# Class mapping
CLASS_MAP = {0: "Human", 1: "Animal"}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}

# Function to process single video
def process_video(video_path):
    vid_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out_det = cv2.VideoWriter(os.path.join(DETECTOR_OUTPUT, vid_name),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    out_clf = cv2.VideoWriter(os.path.join(CLASSIFIER_OUTPUT, vid_name),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), desc=f"Processing {vid_name}"):
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = torchvision.transforms.functional.to_tensor(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            detections = detector(img_tensor)[0]

        det_frame = frame.copy()
        clf_frame = frame.copy()

        for box, label, score in zip(detections["boxes"], detections["labels"], detections["scores"]):
            if score < CONF_THRESH:
                continue

            x1, y1, x2, y2 = [int(b) for b in box]

            # Detector only box
            cv2.rectangle(det_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Classifier
            cropped = pil_img.crop((x1, y1, x2, y2))
            input_clf = clf_transform(cropped).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out_cls = classifier(input_clf)
                pred = torch.argmax(out_cls, dim=1).item()

            cls_name = CLASS_MAP[pred]
            color = COLORS[pred]
            cv2.rectangle(clf_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(clf_frame, f"{cls_name} {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_det.write(det_frame)
        out_clf.write(clf_frame)

    cap.release()
    out_det.release()
    out_clf.release()
    print(f"{vid_name} processed ✅ Detector saved, Classifier saved")

# Process all videos in input folder
video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".mp4", ".avi", ".mov"))]
for vid in video_files:
    process_video(os.path.join(INPUT_FOLDER, vid))

print("All videos processed successfully ✅")
