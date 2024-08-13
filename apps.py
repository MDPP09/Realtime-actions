import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile
import tensorflow as tf
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load your pre-trained model
model = tf.keras.models.load_model('ActionModel.keras')

# Define the actions
actions = ['HALO', 'SEHAT', 'TERIMAKASIH', 'NAMA', 'KAMUUGANTENG', 'WAHKEREN', 'SAMA-SAMA']

# Colors for visualization
colors = [
    (245, 117, 16), (117, 245, 16), (16, 117, 245),
    (245, 16, 117), (16, 245, 117), (117, 16, 245),
    (245, 245, 16)
]

# MediaPipe model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

def extract_keypoints(results):
    def landmarks_to_array(landmarks):
        if landmarks is None:
            return np.zeros((1662,))
        return np.array([
            [kp.x for kp in landmarks],
            [kp.y for kp in landmarks],
            [kp.z for kp in landmarks]
        ]).flatten()

    pose_landmarks = landmarks_to_array(results.pose_landmarks.landmark if results.pose_landmarks else [])
    face_landmarks = landmarks_to_array(results.face_landmarks.landmark if results.face_landmarks else [])
    left_hand_landmarks = landmarks_to_array(results.left_hand_landmarks.landmark if results.left_hand_landmarks else [])
    right_hand_landmarks = landmarks_to_array(results.right_hand_landmarks.landmark if results.right_hand_landmarks else [])

    keypoints_array = np.concatenate([pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks])

    if keypoints_array.size < 1662:
        keypoints_array = np.concatenate([keypoints_array, np.zeros(1662 - keypoints_array.size)])

    return keypoints_array

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.threshold = 0.5

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")  # Convert frame to BGR
        image, results = mediapipe_detection(image, mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5))
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        if len(self.sequence) == 30:
            res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            if res[np.argmax(res)] > self.threshold:
                if not self.sentence or actions[np.argmax(res)] != self.sentence[-1]:
                    self.sentence.append(actions[np.argmax(res)])

            if len(self.sentence) > 5:
                self.sentence = self.sentence[-5:]

            detected_action = actions[np.argmax(res)] if res[np.argmax(res)] > self.threshold else "No Detection"
            st.write(f"Detected Action: {detected_action} ({res[np.argmax(res)]:.2f})")

        return image
def process_video(file):
    sequence = []
    sentence = []
    threshold = 0.5

    cap = cv2.VideoCapture(file)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if res[np.argmax(res)] > threshold:
                    if not sentence or actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                detected_action = actions[np.argmax(res)] if res[np.argmax(res)] > threshold else "No Detection"
                st.write(f"Detected Action: {detected_action} ({res[np.argmax(res)]:.2f})")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, channels="RGB")

            # Display the detected actions in the sidebar
            st.sidebar.subheader("Detection Results")
            st.sidebar.write(' '.join(sentence))

    cap.release()

def process_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(image_rgb, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)

        # Create a sequence of 30 frames with the same keypoints
        sequence = [keypoints] * 30

        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        action = actions[np.argmax(res)] if res[np.argmax(res)] > 0.5 else "No Detection"

        st.image(image, channels="BGR")
        st.write(f"Detected Action: {action} ({res[np.argmax(res)]:.2f})")

# Streamlit UI
st.sidebar.markdown('<h2 style="font-size:20px;">Realtime-Action Detection</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:14px;">By Revan</p>', unsafe_allow_html=True)
st.sidebar.image('https://www.pngkey.com/png/detail/268-2686866_logo-gundar-universitas-gunadarma-logo-png.png', caption='Gunadarma', use_column_width=True)

option = st.selectbox("Select Input Type", ("Webcam", "Upload Image", "Upload Video"))

if option == "Webcam":
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
