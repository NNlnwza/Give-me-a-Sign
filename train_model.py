import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

def load_data():
    base_dir = "dataset"
    X = []  # ข้อมูล landmarks
    y = []  # labels
    
    # โหลดข้อมูลจากทุกโฟลเดอร์
    for gesture_name in os.listdir(base_dir):
        gesture_dir = os.path.join(base_dir, gesture_name)
        if os.path.isdir(gesture_dir):
            for file in os.listdir(gesture_dir):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(gesture_dir, file))
                    X.append(data)
                    y.append(gesture_name)
    
    return np.array(X), np.array(y)

def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    # โหลดข้อมูล
    X, y = load_data()
    
    # แปลง labels เป็น one-hot encoding
    unique_labels = np.unique(y)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[label] for label in y])
    y_one_hot = to_categorical(y_encoded, num_classes=len(unique_labels))
    
    # แบ่งข้อมูลเป็น training และ testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    # สร้างและเทรนโมเดล
    model = create_model(input_shape=(X.shape[1],), num_classes=len(unique_labels))
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    # บันทึกโมเดลและ label mapping
    model.save('hand_gesture_model.h5')
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(label_to_idx, f)
    
    print("บันทึกโมเดลและ label mapping เรียบร้อยแล้ว")
    print(f"จำนวนท่าทางที่เทรน: {len(unique_labels)}")
    print(f"ท่าทางทั้งหมด: {', '.join(unique_labels)}")

if __name__ == "__main__":
    train_model()