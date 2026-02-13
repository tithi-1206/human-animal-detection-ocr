import streamlit as st
import os
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import numpy as np
import tempfile
from pathlib import Path
import sys

# Skip online model source check for PaddleOCR (offline-friendly)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

try:
    from paddleocr import PaddleOCR
except ImportError:
    st.error("PaddleOCR not installed. Please run: pip install paddleocr")
    st.stop()

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Human/Animal Detection & OCR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CONSTANTS & CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESH = 0.5
CLASS_MAP = {0: "Human", 1: "Animal"}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}
OCR_LANG = 'en'

# Paths for models
DETECTOR_MODEL_PATH = "models/detector/fasterrcnn_4gb_epoch_10.pth"
CLASSIFIER_MODEL_PATH = "models/classifier/best_classifier.pth"
# HELPER FUNCTIONS - DETECTOR & CLASSIFIER

@st.cache_resource
def load_detector():
    """Load the object detection model"""
    try:
        detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        num_classes = 4  # include background
        in_features = detector.roi_heads.box_predictor.cls_score.in_features
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        detector.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=DEVICE))
        detector.to(DEVICE)
        detector.eval()
        return detector
    except Exception as e:
        st.error(f"Error loading detector: {str(e)}")
        return None

@st.cache_resource
def load_classifier():
    """Load the classification model"""
    try:
        classifier = torchvision.models.efficientnet_b0(weights=None)
        in_features = classifier.classifier[1].in_features
        classifier.classifier[1] = torch.nn.Linear(in_features, 2)
        classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE))
        classifier.to(DEVICE)
        classifier.eval()
        return classifier
    except Exception as e:
        st.error(f"Error loading classifier: {str(e)}")
        return None

def get_classifier_transform():
    """Get the transform for classifier input"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def process_video_stream(video_path, detector, classifier, progress_bar, status_text):
    """Process video and return paths to output videos"""
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output files
    temp_dir = tempfile.mkdtemp()
    detector_output = os.path.join(temp_dir, "detector_output.mp4")
    classifier_output = os.path.join(temp_dir, "classifier_output.mp4")
    
    # Create video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_det = cv2.VideoWriter(detector_output, fourcc, fps, (width, height))
    out_clf = cv2.VideoWriter(classifier_output, fourcc, fps, (width, height))
    
    clf_transform = get_classifier_transform()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        # Convert to PIL
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = torchvision.transforms.functional.to_tensor(pil_img).unsqueeze(0).to(DEVICE)
        
        # Run detector
        with torch.no_grad():
            detections = detector(img_tensor)[0]
        
        det_frame = frame.copy()
        clf_frame = frame.copy()
        
        # Process each detection
        for box, label, score in zip(detections["boxes"], detections["labels"], detections["scores"]):
            if score < CONF_THRESH:
                continue
            
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Detector output (yellow boxes)
            cv2.rectangle(det_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(det_frame, f"Det: {score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Classifier output
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
    
    return detector_output, classifier_output

# HELPER FUNCTIONS - OCR
@st.cache_resource
def load_ocr_model():
    """Load PaddleOCR model"""
    try:
        ocr = PaddleOCR(
            use_textline_orientation=True,
            lang=OCR_LANG,
            use_gpu=False,
            show_log=False
        )
        return ocr
    except Exception as e:
        st.error(f"Error loading OCR model: {str(e)}")
        return None

def enhance_image(img):
    """Enhance image for better OCR"""

    # CLAHE contrast boost
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

def draw_ocr_result(img, result):
    """Draw OCR results on image"""
    if not result or not result[0]:
        return img
    
    for line in result[0]:
        if line is None:
            continue
        box = line[0]
        txt, score = line[1]
        
        # Draw polygon box
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Put text
        x, y = int(box[0][0]), int(box[0][1]) - 10
        cv2.putText(img, f"{txt} ({score:.2f})", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return img

def process_ocr_image(image, ocr_model):
    """Process image with OCR"""
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Enhance
    enhanced = enhance_image(img_cv)
    
    # Run OCR
    result = ocr_model.ocr(enhanced, cls=True)
    
    # Extract text
    extracted_lines = []
    if result and result[0]:
        for line in result[0]:
            if line and line[1]:
                text, conf = line[1]
                extracted_lines.append(f"{text} (confidence: {conf:.3f})")
    
    full_text = "\n".join(extracted_lines) if extracted_lines else "[No text detected]"
    
    # Draw results
    visualized = draw_ocr_result(enhanced.copy(), result)
    visualized_rgb = cv2.cvtColor(visualized, cv2.COLOR_BGR2RGB)
    
    return full_text, visualized_rgb

# MAIN APP
def main():
    # Header
    st.title("🔍 AI Detection & OCR System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        st.info(f"**Device:** {DEVICE}")
        st.info(f"**Detection Confidence:** {CONF_THRESH}")
        
        st.markdown("---")
        st.header("📋 Instructions")
        st.markdown("""
        **Detection & Classification:**
        - Upload a video file
        - Get detector and classifier outputs
        
        **OCR:**
        - Upload an image with stenciled text
        - Get extracted text and visualization
        """)
    
    # Main selection
    tab1, tab2 = st.tabs(["🎥 Human & Animal Detection", "📝 OCR for Stenciled Text"])
    
    # TAB 1: DETECTION & CLASSIFICATION
    with tab1:
        st.header("Human & Animal Detection System")
        st.markdown("Upload a video to detect and classify humans and animals")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov"],
            key="video_uploader"
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_video.name)
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())
            
            # Display uploaded video
            st.video(temp_video_path)
            
            # Process button
            if st.button("🚀 Process Video", type="primary", key="process_video"):
                with st.spinner("Loading models..."):
                    detector = load_detector()
                    classifier = load_classifier()
                
                if detector is None or classifier is None:
                    st.error("Failed to load models. Please check model paths.")
                    return
                
                st.success("✅ Models loaded successfully!")
                
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video
                try:
                    detector_out, classifier_out = process_video_stream(
                        temp_video_path, 
                        detector, 
                        classifier, 
                        progress_bar, 
                        status_text
                    )
                    
                    status_text.text("✅ Processing complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Detector Output (Yellow Boxes)**")
                        st.video(detector_out)
                        with open(detector_out, "rb") as f:
                            st.download_button(
                                "⬇️ Download Detector Output",
                                f,
                                file_name="detector_output.mp4",
                                mime="video/mp4"
                            )
                    
                    with col2:
                        st.markdown("**Classifier Output (Green=Human, Red=Animal)**")
                        st.video(classifier_out)
                        with open(classifier_out, "rb") as f:
                            st.download_button(
                                "⬇️ Download Classifier Output",
                                f,
                                file_name="classifier_output.mp4",
                                mime="video/mp4"
                            )
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.exception(e)
    
    # TAB 2: OCR
    with tab2:
        st.header("OCR for Stenciled / Industrial Text")
        st.markdown("Upload an image with stenciled or painted text")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            key="image_uploader"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process button
            if st.button("🔍 Extract Text", type="primary", key="process_ocr"):
                with st.spinner("Loading OCR model..."):
                    ocr_model = load_ocr_model()
                
                if ocr_model is None:
                    st.error("Failed to load OCR model.")
                    return
                
                st.success("✅ OCR model loaded successfully!")
                
                # Process image
                with st.spinner("Processing image..."):
                    try:
                        extracted_text, visualized = process_ocr_image(image, ocr_model)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Extracted Text:**")
                            st.text_area(
                                "Text Output",
                                value=extracted_text,
                                height=300,
                                key="ocr_output"
                            )
                            
                            # Copy to clipboard button
                            if st.button("📋 Copy to Clipboard"):
                                st.code(extracted_text)
                                st.success("Text displayed above - you can manually copy it")
                        
                        with col2:
                            st.markdown("**Visualization:**")
                            st.image(visualized, caption="OCR Result with Bounding Boxes", use_container_width=True)
                            
                            # Download button
                            result_img = Image.fromarray(visualized)
                            temp_img_path = os.path.join(tempfile.gettempdir(), "ocr_result.jpg")
                            result_img.save(temp_img_path)
                            with open(temp_img_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download Visualization",
                                    f,
                                    file_name="ocr_result.jpg",
                                    mime="image/jpeg"
                                )
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.exception(e)
    

if __name__ == "__main__":
    main()