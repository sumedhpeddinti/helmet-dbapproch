import sys
import os
import pickle
import cv2
import numpy as np
import torch
import time
import logging
import requests
import base64
from threading import Thread
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from ultralytics.utils import SETTINGS

# Redirect Ultralytics runs dir to temp to avoid clutter
TEMP_RUNS_DIR = "/tmp/ultralytics_noruns"
os.makedirs(TEMP_RUNS_DIR, exist_ok=True)
SETTINGS.update({"runs_dir": TEMP_RUNS_DIR})

# === CONFIGURATION ===
MODEL_PATH = "best.pt"  # single-class helmet model
FACE_DB_PATH = "face_db_converted.pkl"
SIM_THRESHOLD = 0.35
HELMET_CLASS_NAME = "helmet"
VIOLATION_TRIGGER_SECONDS = 5
LOG_LEVEL = logging.INFO
DISABLED_CLASSES = [2]  # Disable person class (class 2)

# Cloudinary configuration
CLOUDINARY_CLOUD_NAME = "dmjbzodmp"
CLOUDINARY_API_KEY = "635179868661266"
CLOUDINARY_API_SECRET = "1YrX3nOQTiLjtMOImmi3a91zdb4"

# Supabase configuration
SUPABASE_URL = "https://uqucdxrmgladlptuuary.supabase.co/functions/v1/challan-api"
SUPABASE_API_KEY = "c4cc9f91b751152111216c30f2985750a007ca2ce5bc4b34a69d0b9186dee805"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=LOG_LEVEL,
)

# Speed boost for GPU convs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def load_face_db(path):
    if not os.path.exists(path):
        logging.warning(f"Face DB missing at {path}")
        return None, None
    with open(path, "rb") as f:
        db = pickle.load(f)
    embeddings = np.array(db.get("embeddings"))
    labels = db.get("labels") or db.get("names")
    if embeddings.size == 0 or not labels:
        logging.warning("Face DB is empty")
        return None, None
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = embeddings / norms
    logging.info(f"Loaded {len(embeddings)} known faces from DB")
    return embeddings, labels


def cosine_match(face_embedding, db_embeddings, labels, thresh=SIM_THRESHOLD):
    if db_embeddings is None or labels is None:
        return None
    face_embedding = face_embedding / (np.linalg.norm(face_embedding) + 1e-9)
    sims = db_embeddings @ face_embedding
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= thresh:
        return labels[best_idx]
    return None


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def upload_to_cloudinary(image_frame):
    """Upload image to Cloudinary and return the URL"""
    try:
        import hashlib
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image_frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare Cloudinary upload URL
        upload_url = f"https://api.cloudinary.com/v1_1/{CLOUDINARY_CLOUD_NAME}/image/upload"
        
        # Generate signature for authenticated upload
        timestamp = str(int(time.time()))
        params_to_sign = f"timestamp={timestamp}{CLOUDINARY_API_SECRET}"
        signature = hashlib.sha1(params_to_sign.encode('utf-8')).hexdigest()
        
        # Prepare data with API key authentication
        data = {
            "file": f"data:image/jpeg;base64,{image_base64}",
            "api_key": CLOUDINARY_API_KEY,
            "timestamp": timestamp,
            "signature": signature
        }
        
        # Make request
        response = requests.post(upload_url, data=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            image_url = result.get('secure_url', '')
            logging.info(f"✅ Image uploaded to Cloudinary: {image_url}")
            return image_url
        else:
            logging.error(f"❌ Cloudinary upload failed: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        logging.error(f"❌ Error uploading to Cloudinary: {e}")
        return ""


def create_challan(roll_number, photo_url):
    """Send violation data to Supabase to create a challan"""
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": SUPABASE_API_KEY
        }
        
        payload = {
            "action": "create_challan",
            "roll_number": roll_number,
            "violation_type": "No Helmet",
            "amount": 100,
            "location": "Entrance",
            "photo_url": photo_url
        }
        
        response = requests.post(SUPABASE_URL, json=payload, headers=headers, timeout=10)
        
        logging.info(f"Supabase response: {response.text}")
        
        if response.status_code == 200:
            logging.info(f"✅ Challan created successfully for {roll_number}")
            return True
        else:
            logging.error(f"❌ Challan creation failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logging.error(f"❌ Error creating challan: {e}")
        return False


def process_violation_async(person_name, frame):
    """Process violation in background thread to avoid blocking camera"""
    try:
        # Upload entire frame as violation photo
        logging.info(f"📤 Uploading violation photo for {person_name}...")
        photo_url = upload_to_cloudinary(frame.copy())
        
        if photo_url:
            # Send challan to Supabase
            logging.info(f"📝 Creating challan for {person_name}...")
            create_challan(person_name, photo_url)
        else:
            logging.warning(f"⚠️  Failed to upload image, challan not created for {person_name}")
    except Exception as e:
        logging.error(f"❌ Error processing violation for {person_name}: {e}")


def main():
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    use_half = use_cuda
    logging.info(f"Using device: {device} (half precision: {use_half})")

    # Face DB
    face_db_embeddings, face_db_labels = load_face_db(FACE_DB_PATH)

    # InsightFace (detects faces/heads)
    ctx_id = 0 if use_cuda else -1
    face_app = FaceAnalysis(providers=("CUDAExecutionProvider", "CPUExecutionProvider"))
    face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    # YOLO helmet-only model
    yolo_model = YOLO(MODEL_PATH)
    if use_cuda:
        yolo_model.to(device)
    yolo_model.overrides.update({
        "save": False,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False,
        "project": TEMP_RUNS_DIR,
        "name": "noop",
        "exist_ok": True,
    })
    logging.info("Models loaded")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        sys.exit(1)
    logging.info("Camera opened - press 'q' to quit, 's' to save frame")

    violation_start_times = {}
    challan_sent = {}  # Track which users have had challans sent to avoid duplicates

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read from camera")
                break

            # Helmets via YOLO
            results = yolo_model(
                frame,
                device=device,
                half=use_half,
                verbose=False,
                save=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                project=TEMP_RUNS_DIR,
                name="noop",
                exist_ok=True,
            )[0]
            helmet_boxes = []
            names_map = yolo_model.model.names
            for box in results.boxes:
                cls_id = int(box.cls)
                # Skip disabled classes (e.g., person class)
                if cls_id in DISABLED_CLASSES:
                    continue
                label_name = names_map.get(cls_id, str(cls_id))
                if label_name != HELMET_CLASS_NAME:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                helmet_boxes.append((int(x1), int(y1), int(x2), int(y2), float(box.conf)))

            # Faces via InsightFace (heads)
            faces = face_app.get(frame)
            face_results = []
            for face in faces:
                fb = face.bbox.astype(int)
                face_box = (fb[0], fb[1], fb[2], fb[3])
                emb = face.embedding
                matched_id = cosine_match(emb, face_db_embeddings, face_db_labels)
                person_name = matched_id if matched_id else "Unknown"
                face_results.append({
                    "name": person_name,
                    "box": face_box,
                })

            # Determine helmet vs no-helmet by overlap
            current_time = time.time()
            for fr in face_results:
                x1, y1, x2, y2 = fr["box"]
                person_name = fr["name"]
                has_helmet = False
                for hx1, hy1, hx2, hy2, _ in helmet_boxes:
                    if bbox_iou((x1, y1, x2, y2), (hx1, hy1, hx2, hy2)) > 0.3:
                        has_helmet = True
                        break

                if not has_helmet and person_name != "Unknown":
                    if person_name not in violation_start_times:
                        violation_start_times[person_name] = current_time
                        challan_sent[person_name] = False  # Reset challan status
                        logging.warning(f"⚠️  {person_name} detected WITHOUT helmet")
                    elif current_time - violation_start_times[person_name] >= VIOLATION_TRIGGER_SECONDS:
                        # Only send challan once per continuous violation period
                        if not challan_sent.get(person_name, False):
                            logging.critical(
                                f"🚨 VIOLATION: {person_name} has been without helmet for {VIOLATION_TRIGGER_SECONDS}+ seconds!"
                            )
                            
                            # Process violation in background thread (non-blocking)
                            violation_thread = Thread(target=process_violation_async, args=(person_name, frame), daemon=True)
                            violation_thread.start()
                            challan_sent[person_name] = True
                        
                        # Reset timer to check again after VIOLATION_TRIGGER_SECONDS
                        violation_start_times[person_name] = current_time
                else:
                    # Person is wearing helmet or left the frame
                    violation_start_times.pop(person_name, None)
                    challan_sent.pop(person_name, None)

            # Draw helmets (green)
            for hx1, hy1, hx2, hy2, conf in helmet_boxes:
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 200, 0), 2)
                cv2.putText(frame, f"HELMET {conf:.2f}", (hx1, hy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

            # Draw faces: green if helmet overlaps, red otherwise
            for fr in face_results:
                x1, y1, x2, y2 = fr["box"]
                person_name = fr["name"]
                has_helmet = any(bbox_iou((x1, y1, x2, y2), (hx1, hy1, hx2, hy2)) > 0.3 for hx1, hy1, hx2, hy2, _ in helmet_boxes)
                if has_helmet:
                    color = (0, 200, 0)
                    status = "HAS HELMET"
                else:
                    color = (0, 0, 255)
                    status = f"NO HELMET: {person_name}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, status, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Display
            try:
                cv2.imshow("Helmet Safety Monitor - WITH CHALLAN", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Exiting...")
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"frame_{int(time.time())}.jpg", frame)
                    logging.info("Frame saved")
            except cv2.error:
                pass

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        logging.info("Resources released")


if __name__ == "__main__":
    main()
