import cv2
import numpy as np
import os
import pickle
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import ndimage

class BiometricAuth:
    def __init__(self):
        self.data_dir = 'biometric_data'
        self.fingerprints = []
        self.face_data = []
        self.labels = []
        self.model = None

    def capture_fingerprint(self, user_id):
        # Code for capturing fingerprint goes here
        print(f"Capturing fingerprint for user: {user_id}")
        
    def capture_face(self, user_id):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Face Capture', gray)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.face_data.append(gray)
                self.labels.append(user_id)
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    def save_data(self):
        with open('face_data.pkl', 'wb') as f:
            pickle.dump((self.face_data, self.labels), f)

    def load_data(self):
        if os.path.exists('face_data.pkl'):
            with open('face_data.pkl', 'rb') as f:
                self.face_data, self.labels = pickle.load(f)

    def train_model(self):
        if len(self.face_data) == 0:
            print("No data to train on.")
            return
        x_train, x_test, y_train, y_test = train_test_split(self.face_data, self.labels, test_size=0.2, random_state=42)
        self.model = neighbors.KNeighborsClassifier(n_neighbors=3)
        self.model.fit(np.array(x_train).reshape(len(x_train), -1), y_train)
        y_pred = self.model.predict(np.array(x_test).reshape(len(x_test), -1))
        print(classification_report(y_test, y_pred))

    def authenticate_face(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prediction = self.model.predict(gray.reshape(1, -1))
            cv2.putText(frame, f"Predicted: {prediction[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Authentication', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def capture_and_authenticate(self, user_id):
        self.capture_face(user_id)
        self.save_data()
        self.train_model()
        self.authenticate_face()

if __name__ == "__main__":
    biometric_auth = BiometricAuth()
    user_id_input = input("Enter user ID for registration: ")
    biometric_auth.capture_and_authenticate(user_id_input)