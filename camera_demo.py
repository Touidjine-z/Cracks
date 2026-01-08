"""
Real-time Crack Detection using Webcam
======================================
Captures video from your computer's front camera and detects cracks in real-time.

Usage:
    python camera_demo.py
    
Controls:
    - Press 'q' to quit
    - Press 's' to save current frame with detections
    - Press 'c' to clear detections and continue
"""

import cv2
import numpy as np
from crack_detector_cv import CrackDetectorCV
import os
from datetime import datetime


def main():
    print("=" * 60)
    print("REAL-TIME CRACK DETECTION - WEBCAM")
    print("=" * 60)
    
    # Load the trained model
    model_path = "crack_detector_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Error: Model not found!")
        print("   Please train the model first:")
        print("   python quick_start.py --train")
        return
    
    print("Loading model...")
    detector = CrackDetectorCV(model_path=model_path, stride=8)
    detector.load_model()
    print("✓ Model loaded successfully!")
    
    # Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)  # 0 = default camera
    
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam!")
        print("   Make sure your camera is connected and not in use.")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Webcam opened: {width}x{height}")
    
    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 'c' to clear detections")
    print("  - Press '+' to increase confidence threshold")
    print("  - Press '-' to decrease confidence threshold")
    print("=" * 60)
    
    confidence_threshold = 0.5
    frame_count = 0
    saved_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read frame from webcam")
                break
            
            frame_count += 1
            
            # Process every 3rd frame to improve speed
            if frame_count % 3 == 0:
                # Save frame temporarily
                temp_path = "temp_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Detect cracks
                result = detector.detect_cracks(
                    temp_path,
                    confidence_threshold=confidence_threshold,
                )

                if result is None:
                    detections = []
                else:
                    _, detections = result
                
                # Draw detections on frame
                display_frame = frame.copy()
                for det in detections:
                    x, y, w, h = det['x'], det['y'], det['width'], det['height']
                    confidence = det['confidence']
                    
                    # Draw rectangle
                    color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)  # Green if high confidence, Orange otherwise
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw confidence text
                    label = f"{confidence:.1%}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(display_frame, (x, y - 20), (x + label_size[0], y), color, -1)
                    cv2.putText(display_frame, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add info overlay
                info_text = f"Detections: {len(detections)} | Threshold: {confidence_threshold:.2f} | FPS: ~{30//3}"
                cv2.putText(display_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                # Add controls reminder
                controls = "Q:Quit | S:Save | C:Clear | +/-:Threshold"
                cv2.putText(display_frame, controls, (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(display_frame, controls, (10, height - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Delete temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                display_frame = frame
            
            # Show frame
            cv2.imshow('Crack Detection - Webcam', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n✓ Exiting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"webcam_capture_{timestamp}.jpg"
                cv2.imwrite(save_path, display_frame)
                saved_count += 1
                print(f"✓ Frame saved: {save_path}")
            elif key == ord('c'):
                # Clear/reset
                print("✓ Detections cleared")
            elif key == ord('+') or key == ord('='):
                # Increase threshold
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                print(f"✓ Confidence threshold: {confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                # Decrease threshold
                confidence_threshold = max(0.05, confidence_threshold - 0.05)
                print(f"✓ Confidence threshold: {confidence_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("SESSION SUMMARY:")
        print(f"  - Frames processed: {frame_count}")
        print(f"  - Frames saved: {saved_count}")
        print("=" * 60)
        print("✓ Camera closed successfully")


if __name__ == "__main__":
    main()
