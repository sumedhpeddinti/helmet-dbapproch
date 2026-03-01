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
                        logging.warning(f"⚠️  {person_name} detected WITHOUT helmet")
                    elif current_time - violation_start_times[person_name] >= VIOLATION_TRIGGER_SECONDS:
                        logging.critical(
                            f"🚨 VIOLATION: {person_name} has been without helmet for {VIOLATION_TRIGGER_SECONDS}+ seconds!"
                        )
                        violation_start_times[person_name] = current_time
                else:
                    violation_start_times.pop(person_name, None)

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
                cv2.imshow("Helmet Safety Monitor (best.pt)", frame)
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
