from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS  # เพิ่มการนำเข้า CORS
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
import time
import threading
import json
from PIL import Image
import io
import socket
import os
import cv2
import numpy as np
import mediapipe as mp
import subprocess
from datetime import datetime

app = Flask(__name__)
CORS(app)  # เพิ่มการรองรับ CORS

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

# Global variables for camera control
camera_active = False
camera_lock = threading.Lock()

# เปิดกล้อง
camera = cv2.VideoCapture(0)

# Global variables
current_text = ""
last_gesture = ""
last_gesture_time = 0
last_activity_time = 0
gesture_duration = 1.0
inactivity_timeout = 5.0
sentence = []
max_words = 6

# Load model and setup MediaPipe
# ลบตัวแปร global model, label_mapping, idx_to_label
# ในฟังก์ชัน process_frame() และ generate_frames() ให้โหลดโมเดลและ label mapping ใหม่ทุกครั้งจากไฟล์ hand_gesture_model.h5 และ label_mapping.pkl
# ปรับให้การแปลง label เป็นภาษาไทยใช้ eng_to_thai ทุกครั้งที่ทำนาย

# สร้าง dictionary สำหรับแปลงข้อความ
eng_to_thai = {
    'Sick': 'ไม่สบาย',
    'Miss': 'คิดถึง',
    'Hello': 'สวัสดี',
    'You': 'คุณ',
    'Me': 'ฉัน',
    'Goodluck': 'โชคดี',
    'Like': 'ชอบ',
    'Sad': 'เสียใจ',
    'Thank you': 'ขอบคุณ',
    'Sorry': 'ขอโทษ',
    'Love': 'รัก',
    'Hungry': 'หิว',
    'Thankyou': 'ขอบคุณ'  # เพิ่มรูปแบบการเขียนแบบติดกัน
}

# ตัวแปรสถานะการเก็บข้อมูล
collect_status = {'gesture_name': None, 'count': 0, 'max': 100}

def generate_frames():
    global current_text, last_gesture, last_gesture_time, last_activity_time, sentence
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    
    while True:
        with camera_lock:
            if not camera_active:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                text = "กล้องปิดอยู่"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (640 - text_size[0]) // 2
                text_y = (480 + text_size[1]) // 2
                cv2.putText(img, text, (text_x, text_y),
                           font, font_scale, (255, 255, 255), thickness)
                ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.033)  # ~30 FPS
                continue
                
            success, img = cap.read()
            if not success:
                print("Cannot open camera")
                break
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            current_time = time.time()
            
            # Check for inactivity timeout
            if current_time - last_activity_time > inactivity_timeout:
                sentence = []
            
            if results.multi_hand_landmarks:
                all_landmarks = []
                hand_count = 0
                
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    all_landmarks.extend(landmarks)
                    hand_count += 1
                
                if hand_count > 0:
                    if hand_count == 1:
                        all_landmarks.extend([0.0] * 63)
                    
                    landmarks = np.array(all_landmarks).reshape(1, -1)
                    # โหลดโมเดลและ label mapping ใหม่ทุกครั้ง
                    model = load_model('hand_gesture_model.h5')
                    with open('label_mapping.pkl', 'rb') as f:
                        label_mapping = pickle.load(f)
                    idx_to_label = {v: eng_to_thai.get(k, k) for k, v in label_mapping.items()}
                    prediction = model.predict(landmarks, verbose=0)
                    predicted_idx = np.argmax(prediction[0])
                    predicted_label = idx_to_label[predicted_idx]
                    
                    if predicted_label == last_gesture:
                        if current_time - last_gesture_time >= gesture_duration:
                            sentence.append(predicted_label)
                            if len(sentence) > max_words:
                                sentence = []
                            last_gesture_time = current_time
                            last_activity_time = current_time
                    else:
                        last_gesture = predicted_label
                        last_gesture_time = current_time
            
            ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_frame(frame):
    global current_text, last_gesture, last_gesture_time, last_activity_time, sentence
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    current_time = time.time()
    
    # Check for inactivity timeout
    if current_time - last_activity_time > inactivity_timeout:
        sentence = []
    
    if results.multi_hand_landmarks:
        all_landmarks = []
        hand_count = 0
        
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            all_landmarks.extend(landmarks)
            hand_count += 1
        
        if hand_count > 0:
            # เพิ่มข้อมูล dummy สำหรับมือที่สองถ้ามีแค่มือเดียว
            if hand_count == 1:
                all_landmarks.extend([0.0] * 63)
            
            landmarks = np.array(all_landmarks).reshape(1, -1)
            # โหลดโมเดลและ label mapping ใหม่ทุกครั้ง
            model = load_model('hand_gesture_model.h5')
            with open('label_mapping.pkl', 'rb') as f:
                label_mapping = pickle.load(f)
            idx_to_label = {v: eng_to_thai.get(k, k) for k, v in label_mapping.items()}
            prediction = model.predict(landmarks, verbose=0)
            predicted_idx = np.argmax(prediction[0])
            predicted_label = idx_to_label[predicted_idx]
            
            if predicted_label == last_gesture:
                if current_time - last_gesture_time >= gesture_duration:
                    if len(sentence) < max_words:
                        sentence.append(predicted_label)
                    else:
                        sentence = [predicted_label]  # รีเซ็ตประโยคและเริ่มใหม่
                    last_gesture_time = current_time
                    last_activity_time = current_time
            else:
                last_gesture = predicted_label
                last_gesture_time = current_time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    file = request.files['frame']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    process_frame(frame)
    return jsonify({'success': True})

@app.route('/get_text')
def get_text():
    if not sentence:
        return jsonify({'text': 'รอการตรวจจับ...', 'last_gesture': last_gesture if last_gesture else ''})
    
    # แปลงข้อความเป็นภาษาไทย
    thai_sentence = []
    for word in sentence:
        thai_word = eng_to_thai.get(word, word)
        thai_sentence.append(thai_word)
    
    return jsonify({
        'text': ' '.join(thai_sentence),
        'last_gesture': eng_to_thai.get(last_gesture, last_gesture) if last_gesture else ''
    })

@app.route('/clear_text')
def clear_text():
    global sentence, last_gesture
    sentence = []
    last_gesture = ''
    return jsonify({'success': True})

@app.route('/start_camera')
def start_camera():
    global camera_active
    with camera_lock:
        camera_active = True
    return jsonify({'success': True})

@app.route('/stop_camera')
def stop_camera():
    global camera_active
    with camera_lock:
        camera_active = False
    return jsonify({'success': True})

@app.route('/start_collect', methods=['POST'])
def start_collect():
    data = request.get_json()
    gesture_name = data.get('gesture_name')
    if not gesture_name:
        return jsonify({'success': False, 'message': 'No gesture name'}), 400
    base_dir = 'dataset'
    gesture_dir = os.path.join(base_dir, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)
    collect_status['gesture_name'] = gesture_name
    collect_status['count'] = 0
    return jsonify({'success': True})

@app.route('/collect_frame', methods=['POST'])
def collect_frame():
    gesture_name = request.form.get('gesture_name')
    if not gesture_name:
        return jsonify({'success': False, 'message': 'No gesture name'}), 400
    if 'frame' not in request.files:
        return jsonify({'success': False, 'message': 'No frame'}), 400
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    all_landmarks = []
    hand_count = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                all_landmarks.extend([landmark.x, landmark.y, landmark.z])
            hand_count += 1
        if hand_count == 1:
            all_landmarks.extend([0.0] * 63)
        if hand_count > 0:
            base_dir = 'dataset'
            gesture_dir = os.path.join(base_dir, gesture_name)
            os.makedirs(gesture_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            np.save(os.path.join(gesture_dir, f'{timestamp}.npy'), all_landmarks)
            collect_status['count'] += 1
            return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'No hand detected'})

@app.route('/train_model', methods=['POST'])
def train_model_api():
    # เรียก train_model.py เป็น subprocess
    try:
        result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # โหลด label mapping ใหม่
            import pickle
            with open('label_mapping.pkl', 'rb') as f:
                label_mapping = pickle.load(f)
            # แปลง label เป็นภาษาไทยถ้าจำเป็น
            gestures = []
            for label in label_mapping.keys():
                if all(ord(c) < 128 for c in label):
                    gestures.append(eng_to_thai.get(label, label))
                else:
                    gestures.append(label)
            return jsonify({'success': True, 'output': result.stdout, 'gestures': gestures})
        else:
            return jsonify({'success': False, 'output': result.stderr})
    except Exception as e:
        return jsonify({'success': False, 'output': str(e)})

@app.route('/get_gestures')
def get_gestures():
    import pickle
    try:
        with open('label_mapping.pkl', 'rb') as f:
            label_mapping = pickle.load(f)
        gestures = []
        for label in label_mapping.keys():
            if all(ord(c) < 128 for c in label):
                gestures.append(eng_to_thai.get(label, label))
            else:
                gestures.append(label)
        return jsonify({'gestures': gestures})
    except Exception:
        return jsonify({'gestures': []})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 