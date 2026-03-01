import sys
import os
import pickle
import cv2
import numpy as np
import torch
import time
import logging
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from ultralytics.utils import SETTINGS

# Push runs_dir to a temp location so Ultralytics won't clutter the project
TEMP_RUNS_DIR = "/tmp/ultralytics_noruns"
os.makedirs(TEMP_RUNS_DIR, exist_ok=True)
SETTINGS.update({"runs_dir": TEMP_RUNS_DIR})

# === CONFIGURATION ===
MODEL_PATH = "hemletYoloV8_100epochs.pt"
FACE_DB_PATH = "face_db_converted.pkl"
SIM_THRESHOLD = 0.35
HEAD_CLASS_NAME = "head"
HELMET_CLASS_NAME = "helmet"
VIOLATION_TRIGGER_SECONDS = 5
LOG_LEVEL = logging.INFO
DISABLED_CLASSES = [2]  # Disable person class (class 2)

# === LOGGING ===
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=LOG_LEVEL
)

# Enable CuDNN autotune for faster convs when on GPU
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
    # L2 normalize for cosine similarity
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


def main():
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    use_half = use_cuda  # half precision only on GPU
    logging.info(f"Using device: {device} (half precision: {use_half})")

    # Load DB
    face_db_embeddings, face_db_labels = load_face_db(FACE_DB_PATH)

    # Load models
    ctx_id = 0 if use_cuda else -1
    face_app = FaceAnalysis(providers=("CUDAExecutionProvider", "CPUExecutionProvider"))
    face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    
    yolo_model = YOLO(MODEL_PATH)
    if use_cuda:
        yolo_model.to(device)
    # Disable Ultralytics auto-saving runs/detect outputs and enforce temp runs_dir
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

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        sys.exit(1)
    logging.info("Camera opened - press 'q' to quit, 's' to save frame")

    violation_start_times = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read from camera")
                break

            # === YOLO DETECTION ===
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
            boxes = results.boxes
            names = yolo_model.model.names

            detections = []
            for box in boxes:
                cls_id = int(box.cls)
                # Skip disabled classes (e.g., person class)
                if cls_id in DISABLED_CLASSES:
                    continue
                label_name = names.get(cls_id, str(cls_id))
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "class": label_name,
                    "conf": conf
                })

            head_detections = [d for d in detections if d["class"] == HEAD_CLASS_NAME]
            helmet_detections = [d for d in detections if d["class"] == HELMET_CLASS_NAME]

            # === FACE RECOGNITION (only on heads) ===
            face_results = []
            if head_detections:
                faces = face_app.get(frame)
                for head_det in head_detections:
                    x1, y1, x2, y2 = head_det["box"]
                    matched_id = None
                    best_iou = 0.0
                    
                    # Find best matching face
                    for face in faces:
                        fb = face.bbox.astype(int)
                        face_box = (fb[0], fb[1], fb[2], fb[3])
                        iou = bbox_iou((x1, y1, x2, y2), face_box)
                        if iou > best_iou:
                            best_iou = iou
                            emb = face.embedding
                            matched_id = cosine_match(emb, face_db_embeddings, face_db_labels)
                    
                    person_name = matched_id if matched_id else "Unknown"
                    face_results.append({
                        "name": person_name,
                        "box": (x1, y1, x2, y2),
                        "iou": best_iou
                    })

            # === VIOLATION TRACKING ===
            current_time = time.time()
            helmet_boxes_set = set()
            for h_det in helmet_detections:
                hx1, hy1, hx2, hy2 = h_det["box"]
                helmet_boxes_set.add((hx1, hy1, hx2, hy2))

            for face_res in face_results:
                person_name = face_res["name"]
                x1, y1, x2, y2 = face_res["box"]
                
                # Check if head has helmet
                has_helmet = False
                for hx1, hy1, hx2, hy2 in helmet_boxes_set:
                    iou = bbox_iou((x1, y1, x2, y2), (hx1, hy1, hx2, hy2))
                    if iou > 0.3:
                        has_helmet = True
                        break

                if not has_helmet and person_name != "Unknown":
                    if person_name not in violation_start_times:
                        violation_start_times[person_name] = current_time
                        logging.warning(f"⚠️  {person_name} detected WITHOUT helmet")
                    elif current_time - violation_start_times[person_name] >= VIOLATION_TRIGGER_SECONDS:
                        logging.critical(f"🚨 VIOLATION: {person_name} has been without helmet for {VIOLATION_TRIGGER_SECONDS}+ seconds!")
                        violation_start_times[person_name] = current_time
                else:
                    violation_start_times.pop(person_name, None)

            # === DRAW DETECTIONS ===
            # Draw helmets (green)
            for det in helmet_detections:
                x1, y1, x2, y2 = det["box"]
                conf = det["conf"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"HELMET {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

            # Draw heads with names (red if no helmet, green if helmet)
            for face_res in face_results:
                x1, y1, x2, y2 = face_res["box"]
                person_name = face_res["name"]
                
                has_helmet = False
                for hx1, hy1, hx2, hy2 in helmet_boxes_set:
                    if bbox_iou((x1, y1, x2, y2), (hx1, hy1, hx2, hy2)) > 0.3:
                        has_helmet = True
                        break
                
                if has_helmet:
                    color = (0, 200, 0)
                    status = "HAS HELMET"
                else:
                    color = (0, 0, 255)
                    status = f"NO HELMET: {person_name}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, status, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # === DISPLAY ===
            try:
                cv2.imshow("Helmet Safety Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Exiting...")
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"frame_{int(time.time())}.jpg", frame)
                    logging.info("Frame saved")
            except cv2.error:
                # No display support - just continue processing
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
