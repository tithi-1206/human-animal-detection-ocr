import os
import sys
import cv2
import numpy as np
from PIL import Image
import pyperclip

# Skip online model source check (offline-friendly)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

try:
    from paddleocr import PaddleOCR
except ImportError as e:
    print("PaddleOCR import failed. Make sure it's installed: pip install paddleocr")
    sys.exit(1)

# CONFIG: Optional local model paths for 100% offline
LOCAL_MODELS = {
    'det_model_dir': None,  
    'rec_model_dir': None, 
    'cls_model_dir': None,  
}

OCR_LANG = 'en'  

def enhance_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

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

def draw_result(img, result):
    """Manual drawing of boxes + text (replacement for draw_ocr)"""
    for line in result:
        if line is None:
            continue
        box = line[0]           # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        txt, score = line[1]    # (text, confidence)

        # Draw polygon box
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Put text above first point
        x, y = int(box[0][0]), int(box[0][1]) - 10
        cv2.putText(img, f"{txt} ({score:.2f})", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img

def run_ocr(image_path):
    print(f"Processing: {os.path.basename(image_path)}")

    enhanced = enhance_image(image_path)

    print("Initializing PaddleOCR (may download models first time)...")
    ocr = PaddleOCR(
        use_textline_orientation=True,   # for handling rotated text lines
        lang=OCR_LANG,
        use_gpu=False,
        **{k: v for k, v in LOCAL_MODELS.items() if v is not None}
    )

    print("Running OCR...")
    result = ocr.ocr(enhanced, cls=True)   # cls=True still ok for classification

    if not result or not result[0]:
        print("→ No text detected.")
        return "[No text recognized]", enhanced

    # Flatten result (PaddleOCR 3.x returns list of lists)
    extracted_lines = []
    for line in result[0]:
        if line and line[1]:
            text, conf = line[1]
            print(f"  {text}  (conf: {conf:.3f})")
            extracted_lines.append(text)

    full_text = "\n".join(extracted_lines).strip()
    if not full_text:
        full_text = "[No readable text]"


    print("\nExtracted text:")
    print("═" * 60)
    print(full_text)
    print("═" * 60)

    return full_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python paddle_ocr_crate_fixed.py your_image.jpg")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    text = run_ocr(filepath)

    try:
        pyperclip.copy(text)
        print("\n→ Copied to clipboard!")
    except Exception as e:
        print(f"Clipboard failed: {e}")