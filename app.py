import os
import pickle
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from keras_facenet import FaceNet
from collections import Counter

# ---------------- LOAD DATA ----------------
feature_list = pickle.load(open('embedding_facenet.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# ---------------- FACE EMBEDDER ----------------
embedder = FaceNet()

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# ---------------- FUNCTIONS ----------------

def save_uploaded_image(uploaded_image):
    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', uploaded_image.name)

        with open(file_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())

        return file_path
    except:
        return None


def detect_and_crop_face(img_path):
    """Detect face using Haar Cascade and return cropped face"""
    img = cv2.imread(img_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None  # No face detected

    # Take the first detected face
    x, y, w, h = faces[0]
    face = img[y:y + h, x:x + w]

    # Convert BGR to RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize for FaceNet (160x160)
    face = cv2.resize(face, (160, 160))

    return face


def extract_features(img_path):
    """Extract FaceNet embedding from an image with face detection"""
    face = detect_and_crop_face(img_path)

    if face is None:
        return None

    # Generate embedding - keras_facenet handles normalization internally
    embedding = embedder.embeddings([face])[0]

    return embedding


def get_actor_name(path):
    return path.split('/')[-2]


def recommend(feature_list, features, filenames, top_k=5):
    similarity = []

    for i in range(len(feature_list)):
        sim = cosine_similarity(
            features.reshape(1, -1),
            feature_list[i].reshape(1, -1)
        )[0][0]
        similarity.append(sim)

    # Top K similar images
    top_indices = np.argsort(similarity)[-top_k:][::-1]

    # Get actor names
    names = [get_actor_name(filenames[i]) for i in top_indices]

    # Voting
    most_common = Counter(names).most_common(1)[0][0]

    return most_common, top_indices


# ---------------- STREAMLIT UI ----------------

st.title("🎬 Which Bollywood Celebrity Are You? (FaceNet)")

uploaded_image = st.file_uploader("Choose an image")

if uploaded_image is not None:

    file_path = save_uploaded_image(uploaded_image)

    if file_path is None:
        st.error("Image upload failed 😕")
    else:

        # Extract features
        features = extract_features(file_path)

        if features is None:
            st.error("❌ No face detected in the image! Please upload a clear photo with a face.")
        else:

            display_image = Image.open(uploaded_image)

            predicted_actor, top_indices = recommend(
                feature_list, features, filenames
            )

            col1, col2 = st.columns(2)

            with col1:
                st.header("Your Uploaded Image")
                st.image(display_image)

            with col2:
                st.header("Seems like: " + predicted_actor)

                st.write("Top matches:")
                for i in top_indices:
                    st.image(filenames[i], width=150)
