import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile
import tensorflow as tf

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


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
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

    # Debug statement to check the size of the keypoints_array
    st.write(f"Size of keypoints array: {keypoints_array.size}")

    if keypoints_array.size < 1662:
        keypoints_array = np.concatenate([keypoints_array, np.zeros(1662 - keypoints_array.size)])

    return keypoints_array


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    height, width, _ = output_frame.shape
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 150), 90 + num * 40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, f'{actions[num]}: {prob:.2f}',
                    (10, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)
    return output_frame


def process_webcam():
    sequence = []
    sentence = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)  # Open the webcam

    stframe = st.empty()  # Placeholder for Streamlit image display

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

            # Debug statement to check the length of the sequence
            st.write(f"Current sequence length: {len(sequence)}")

            if len(sequence) == 30:
                try:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    if res[np.argmax(res)] > threshold:
                        if not sentence or actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    detected_action = actions[np.argmax(res)] if res[np.argmax(res)] > threshold else "No Detection"
                    st.write(f"Detected Action: {detected_action} ({res[np.argmax(res)]:.2f})")

                    image = prob_viz(res, actions, image, colors)
                except IndexError as e:
                    st.error(f"IndexError occurred during model prediction: {str(e)}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            stframe.image(image_rgb, channels="RGB", use_column_width=True)

            # Display the detected actions in the sidebar
            st.sidebar.subheader("Detection Results")
            st.sidebar.write(' '.join(sentence))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def process_image(image):
    sequence = []
    threshold = 0.5

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(image, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)

        # Create a sequence of 30 frames with the same keypoints
        sequence = [keypoints] * 30

        try:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            action = actions[np.argmax(res)] if res[np.argmax(res)] > threshold else "No Detection"

            image = prob_viz(res, actions, image, colors)
            st.image(image, channels="RGB")
            st.write(f"Detected Action: {action} ({res[np.argmax(res)]:.2f})")
        except IndexError as e:
            st.error(f"IndexError occurred during model prediction: {str(e)}")


def process_video(file):
    sequence = []
    sentence = []
    threshold = 0.5

    cap = cv2.VideoCapture(file)

    stframe = st.empty()  # Placeholder for Streamlit image display

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

            # Debug statement to check the length of the sequence
            st.write(f"Current sequence length: {len(sequence)}")

            if len(sequence) == 30:
                try:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    if res[np.argmax(res)] > threshold:
                        if not sentence or actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    detected_action = actions[np.argmax(res)] if res[np.argmax(res)] > threshold else "No Detection"
                    st.write(f"Detected Action: {detected_action} ({res[np.argmax(res)]:.2f})")

                    image = prob_viz(res, actions, image, colors)
                except IndexError as e:
                    st.error(f"IndexError occurred during model prediction: {str(e)}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            stframe.image(image_rgb, channels="RGB", use_column_width=True)

            # Display the detected actions in the sidebar
            st.sidebar.subheader("Detection Results")
            st.sidebar.write(' '.join(sentence))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Streamlit UI
st.sidebar.markdown('<h2 style="font-size:20px;">Realtime-Action detection</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:14px;">By revan</p>', unsafe_allow_html=True)
st.sidebar.image('https://www.pngkey.com/png/full/233-2332677_image-500580-placeholder-transparent.png', use_column_width=True)

st.title('Action Detection')

input_source = st.sidebar.selectbox("Choose Input Source", ["Webcam", "Image", "Video"])

if input_source == "Webcam":
    if st.sidebar.button("Start Webcam"):
        process_webcam()
elif input_source == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        process_image(image)
elif input_source == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        process_video(tfile.name)
