import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import time

# Initialize MediaPipe with improved tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # เปลี่ยนเป็น 2 เพื่อตรวจจับมือทั้งสองข้าง
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

# Create directories for data storage
def create_directories():
    base_dir = "dataset"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def countdown(img, count):
    # วาดวงกลมแสดงการนับถอยหลัง
    center = (img.shape[1]//2, img.shape[0]//2)
    radius = 50
    thickness = 3
    color = (0, 255, 0)
    
    # วาดวงกลม
    cv2.circle(img, center, radius, color, thickness)
    
    # แสดงตัวเลข
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    text = str(count)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = center[0] - text_size[0]//2
    text_y = center[1] + text_size[1]//2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

def collect_data():
    base_dir = create_directories()
    cap = cv2.VideoCapture(0)
    
    print("Please enter the gesture name to collect data:")
    gesture_name = input()
    
    gesture_dir = os.path.join(base_dir, gesture_name)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)
    
    count = 0
    max_images = 500  # จำนวนภาพที่ต้องการเก็บ
    
    print("Get ready! Starting countdown...")
    time.sleep(1)  # รอ 1 วินาทีก่อนเริ่มนับถอยหลัง
    
    while count < max_images:
        success, img = cap.read()
        if not success:
            print("Cannot open camera")
            break
            
        # Flip image horizontally
        img = cv2.flip(img, 1)
            
        # Convert image to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            # เก็บข้อมูล landmarks ของมือทั้งสองข้าง
            all_landmarks = []
            hand_count = 0
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks with different colors for each hand
                color = (255, 192, 203) if hand_count == 0 else (192, 255, 203)  # สีชมพูสำหรับมือซ้าย, สีเขียวสำหรับมือขวา
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=color, thickness=2)
                )
                
                # Draw bounding box with different colors
                landmarks_array = np.array([[landmark.x * img.shape[1], landmark.y * img.shape[0]] 
                                         for landmark in hand_landmarks.landmark])
                x_min, y_min = np.min(landmarks_array, axis=0)
                x_max, y_max = np.max(landmarks_array, axis=0)
                cv2.rectangle(img, 
                            (int(x_min-20), int(y_min-20)), 
                            (int(x_max+20), int(y_max+20)), 
                            (0, 0, 255) if hand_count == 0 else (0, 255, 0), 2)
                
                # เก็บข้อมูล landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                all_landmarks.extend(landmarks)
                hand_count += 1
            
            # ถ้ามีมืออย่างน้อย 1 ข้าง
            if hand_count > 0:
                # ถ้ามีมือแค่ข้างเดียว ให้เพิ่มข้อมูลว่างสำหรับมือที่ขาดหายไป
                if hand_count == 1:
                    # เพิ่มข้อมูลว่างสำหรับมือที่สอง (63 ค่า: 21 จุด × 3 ค่าต่อจุด)
                    all_landmarks.extend([0.0] * 63)
                
                # บันทึกข้อมูล
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                np.save(os.path.join(gesture_dir, f"{timestamp}.npy"), all_landmarks)
                count += 1
                print(f"Data saved: {count}/{max_images}")
        
        # แสดงภาพ
        cv2.putText(img, f"Images captured: {count}/{max_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Gesture Data Collection", img)
        
        # หยุดเมื่อถ่ายครบ 300 รูป
        if count >= max_images:
            print(f"Completed collecting {max_images} images for gesture: {gesture_name}")
            break
            
        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data() 