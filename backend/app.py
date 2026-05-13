import cv2
import numpy as np
import datetime
import os
import time
import math
from flask import Flask, render_template, Response, jsonify

# Conditional imports
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    HAS_ML = True
except ImportError:
    HAS_ML = False

# Initialize Flask with custom paths to match new structure
app = Flask(__name__, 
            template_folder='../frontend/templates', 
            static_folder='../frontend/static')

# --- Configuration & Model Loading ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
FACE_PROTO = os.path.join(BASE_PATH, "face_detector/deploy.prototxt")
FACE_MODEL = os.path.join(BASE_PATH, "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
MASK_MODEL = os.path.join(BASE_PATH, "mask_detector.model")

# Global state
stats = {
    "total_people": 0,
    "with_mask": 0,
    "without_mask": 0,
    "partial_mask": 0,
    "safety_score": 100,
    "start_time": datetime.datetime.now(),
    "uptime": "00:00:00"
}

# Thresholds
MIN_FACE_CONFIDENCE = 0.15
MIN_FACE_SIZE = 50
PARTIAL_COVER_THRESHOLD = 0.2

# Load Models
faceNet = None
maskNet = None
# Fallback Cascade Detector
haarNet = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if HAS_ML:
    try:
        if os.path.exists(FACE_PROTO) and os.path.exists(FACE_MODEL):
            faceNet = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
            print("Successfully loaded Caffe Face Detector")
        else:
            print("Warning: Caffe Face Detector files missing. Using Haar Cascade fallback.")
            
        if os.path.exists(MASK_MODEL):
            maskNet = load_model(MASK_MODEL)
            print("Successfully loaded Mask Classifier")
        else:
            print("Warning: Mask Classifier model missing. Classification will be simulated.")
    except Exception as e:
        print(f"Error loading models: {e}")

def calculate_safety_score():
    if stats["total_people"] == 0:
        return 100
    score = (stats["with_mask"] * 100 + stats["partial_mask"] * 50) / stats["total_people"]
    return int(score)

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

face_history = []

def detect_and_predict_mask(frame):
    (h, w) = frame.shape[:2]
    locs = []
    preds = []

    # Try Caffe Detector first
    if faceNet is not None:
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > MIN_FACE_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                
                if (endX - startX) >= MIN_FACE_SIZE and (endY - startY) >= MIN_FACE_SIZE:
                    locs.append((startX, startY, endX, endY))
    
    # Removed Haar Cascade fallback to prevent false positives on background textures

    # Process faces for mask prediction
    for (startX, startY, endX, endY) in locs:
        if maskNet is not None:
            try:
                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (224, 224))
                face = img_to_array(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face = preprocess_input(face)
                pred = maskNet.predict(np.expand_dims(face, axis=0))[0]
                mask_prob, no_mask_prob = pred[0], pred[1]
                
                if abs(mask_prob - no_mask_prob) < PARTIAL_COVER_THRESHOLD:
                    preds.append({'label': "Partial Cover", 'prob': max(mask_prob, no_mask_prob)})
                elif mask_prob > no_mask_prob:
                    preds.append({'label': "Mask", 'prob': mask_prob})
                else:
                    preds.append({'label': "No Mask", 'prob': no_mask_prob})
            except:
                preds.append({'label': "Partial Cover", 'prob': 0.5})
        else:
            # --- ENHANCED SIMULATION FOR BLACK/WHITE MASKS, KERCHIEF AND HAND DETECTION ---
            face_roi = frame[startY:endY, startX:endX]
            if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                # Focus on the lower half of the face where masks/hands/kerchiefs are
                lower_half = face_roi[int(face_roi.shape[0]/2):, :]
                
                # Analyze color and texture
                gray_lower = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_lower, 30, 100) # Slightly more sensitive
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1] + 1e-6)
                
                avg_color = np.mean(lower_half)
                std_color = np.std(lower_half)
                
                # Convert to HSV to detect skin tones (hands)
                hsv_lower = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)
                avg_hsv = np.mean(hsv_lower, axis=(0,1))
                # Typical skin tone range in HSV: H: 0-25, S: 20-150
                is_skin_tone = 0 <= avg_hsv[0] <= 25 and 40 <= avg_hsv[1] <= 150

                # 1. Check for Hand/Skin Tone (Skin color range)
                # If it's skin tone, it's either NO mask or a hand covering the face
                if is_skin_tone:
                    # Hand covering often has more texture/edges (fingers, palm lines)
                    if edge_density > 0.05 or std_color > 40:
                        preds.append({'label': "Partial Cover", 'prob': 0.88})
                    else:
                        preds.append({'label': "No Mask", 'prob': 0.92})
                
                # 2. Check for White/Bright Mask (Low saturation, high brightness)
                elif avg_hsv[1] < 45 and avg_hsv[2] > 120 and std_color < 65:
                    preds.append({'label': "Mask", 'prob': 0.96})
                
                # 3. Check for Black Mask (Low brightness, low texture)
                elif avg_color < 70 and std_color < 40:
                    preds.append({'label': "Mask", 'prob': 0.94})
                
                # 4. Check for Kerchief/Fabric Pattern (High texture/variation)
                elif edge_density > 0.12 or std_color > 55:
                    preds.append({'label': "Partial Cover", 'prob': 0.75})
                
                # 5. Default to No Mask
                else:
                    preds.append({'label': "No Mask", 'prob': 0.92})
            else:
                preds.append({'label': "No Mask", 'prob': 0.92})

    return (locs, preds)

def generate_frames():
    global face_history
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            # Simulation loop if camera fails
            sim_frame = np.zeros((450, 600, 3), dtype=np.uint8)
            cv2.putText(sim_frame, "OFFLINE: Webcam Not Found", (50, 225), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', sim_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
        
        frame = cv2.resize(frame, (600, 450))
        (locs, preds) = detect_and_predict_mask(frame)

        current_faces = []
        for box, pred in zip(locs, preds):
            for hist in face_history:
                if calculate_iou(box, hist['box']) > 0.4:
                    if hist['label'] != pred['label'] and pred['prob'] < 0.7:
                         pred['label'] = hist['label']
                         pred['prob'] = hist['prob']
                    break
            current_faces.append({'box': box, 'label': pred['label'], 'prob': pred['prob']})
        face_history = current_faces

        stats["total_people"] = len(locs)
        stats["with_mask"] = 0
        stats["partial_mask"] = 0
        no_mask_count = 0

        for face in current_faces:
            (startX, startY, endX, endY) = face['box']
            label = face['label']
            prob = face['prob']

            if label == "Mask":
                display_label = f"Mask: {prob*100:.0f}% (Safe)"
                color = (0, 255, 0)
                stats["with_mask"] += 1
            elif label == "Partial Cover":
                display_label = f"Partial Cover: {prob*100:.0f}% (Moderate)"
                color = (0, 255, 255)
                stats["partial_mask"] += 1
            else:
                display_label = f"No Mask: {prob*100:.0f}% (Unsafe)"
                color = (0, 0, 255)
                no_mask_count += 1

            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # --- TOP ALERT MESSAGE ---
            # Draw an alert badge above the head if not wearing mask properly
            # Moved slightly lower to avoid status bar overlap
            y_badge = max(110, startY - 45) 
            if label != "Mask":
                alert_text = "VIOLATION!" if label == "No Mask" else "IMPROPER MASK"
                # Badge background
                badge_w = 140 if label == "No Mask" else 180
                cv2.rectangle(frame, (startX, y_badge), (startX + badge_w, y_badge + 25), color, -1)
                cv2.putText(frame, alert_text, (startX + 5, y_badge + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                # Confirm face recognized and safe
                cv2.rectangle(frame, (startX, y_badge), (startX + 160, y_badge + 25), (0, 150, 0), -1)
                cv2.putText(frame, "FACE RECOGNIZED", (startX + 5, y_badge + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Main Label (Percentage)
            cv2.putText(frame, display_label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        stats["without_mask"] = no_mask_count
        stats["safety_score"] = calculate_safety_score()
        stats["uptime"] = str(datetime.datetime.now() - stats["start_time"]).split(".")[0]

        # --- Dynamic Count-Based Alert Message ---
        if stats["total_people"] == 0:
            alert_msg = None
        elif no_mask_count == 0 and stats["partial_mask"] == 0:
            alert_msg = None
        elif no_mask_count > 0:
            alert_msg = f"ALERT: {no_mask_count} person(s) not wearing mask!"
        elif stats["partial_mask"] > 0:
            alert_msg = f"WARNING: {stats['partial_mask']} person(s) wearing mask improperly"
        else:
            alert_msg = None

        stats["alert_message"] = alert_msg if alert_msg else ""

        # Render alert on video frame
        if alert_msg:
            # Blinking background strip
            current_time = time.time()
            if int(current_time * 2) % 2:
                alert_bg = (0, 0, 180) if "ALERT" in alert_msg else (0, 180, 180)
                cv2.rectangle(frame, (0, 405), (600, 450), alert_bg, -1)
            cv2.putText(frame, alert_msg, (10, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Dashboard Status Overlay (top bar)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (590, 105), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        current_time = time.time()
        safety_score = stats["safety_score"]
        status = "SAFE ZONE" if safety_score >= 90 else "CAUTION" if safety_score >= 70 else "DANGER"
        status_color = (0, 255, 0) if safety_score >= 90 else (0, 255, 255) if safety_score >= 70 else (0, 0, 255)

        if status == "DANGER" and int(current_time * 2) % 2:
            status_color = (255, 255, 255)

        cv2.putText(frame, f"STATUS: {status}  ({safety_score}%)", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(frame,
                    f"Total: {stats['total_people']} | Mask: {stats['with_mask']} | Partial: {stats['partial_mask']} | No: {no_mask_count}",
                    (15, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    stats["uptime"] = str(datetime.datetime.now() - stats["start_time"]).split(".")[0]
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
# Trigger reload for new models
