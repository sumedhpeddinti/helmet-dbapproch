#!/usr/bin/env python3
"""
Simple camera-based YOLO detection with person class disabled
"""

import cv2
import torch
from ultralytics import YOLO

# Configuration
MODEL_PATH = "hemletYoloV8_100epochs.pt"
DISABLED_CLASSES = [2]  # Disable person class

def main():
    # Check CUDA availability
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    print(f"Using device: {device}")
    
    if not use_cuda:
        print("WARNING: CUDA not available! Running on CPU")
    
    # Load YOLO model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to(device)
    
    # Get class names
    class_names = model.names
    print(f"Model classes: {class_names}")
    print(f"Disabled classes: {[class_names.get(i, str(i)) for i in DISABLED_CLASSES]}")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    print("Camera opened successfully!")
    print("Press 'q' to quit, 's' to save frame")
    print("-" * 60)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame")
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = model(frame, device=device, verbose=False)[0]
            
            # Process detections
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls)
                
                # Skip disabled classes
                if cls_id in DISABLED_CLASSES:
                    continue
                
                class_name = class_names.get(cls_id, str(cls_id))
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detections.append({
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
                
                # Draw bounding box with color based on class
                if class_name.lower() == "helmet":
                    color = (0, 0, 255)  # Red for helmet
                elif class_name.lower() == "head":
                    color = (0, 255, 0)  # Green for head
                else:
                    color = (255, 0, 255)  # Magenta for other classes
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with class name and confidence
                label = f"{class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Print detections to terminal
            if detections:
                print(f"\nFrame {frame_count}: {len(detections)} detection(s)")
                for i, det in enumerate(detections, 1):
                    print(f"  [{i}] Class: {det['class_name']} (ID: {det['class_id']}) | "
                          f"Confidence: {det['confidence']:.3f} | "
                          f"BBox: {det['bbox']}")
            
            # Display frame
            cv2.imshow("YOLO Detection - Press 'q' to quit", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord('s'):
                filename = f"frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

if __name__ == "__main__":
    main()
