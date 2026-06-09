import pickle
import numpy as np
import cv2

from PIL import Image
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings and filenames
feature_list = pickle.load(open('embedding_facenet.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load FaceNet
embedder = FaceNet()

# Load test image
img = cv2.imread('sample/alia_bhatt.png')

# Face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

x, y, w, h = faces[0]

face = img[y:y+h, x:x+w]

# Convert BGR to RGB
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

# Resize for FaceNet
face = cv2.resize(face, (160, 160))

# Generate embedding
result = embedder.embeddings([face])[0]

# Similarity search
similarity = []

for i in range(len(feature_list)):
    similarity.append(
        cosine_similarity(
            result.reshape(1, -1),
            feature_list[i].reshape(1, -1)
        )[0][0]
    )

# Best match
index_pos = sorted(
    list(enumerate(similarity)),
    reverse=True,
    key=lambda x: x[1]
)[0][0]

print("Prediction:", filenames[index_pos])

# Top 5 matches
print("\nTop 5 Matches:\n")

top5 = sorted(
    list(enumerate(similarity)),
    reverse=True,
    key=lambda x: x[1]
)[:5]

for idx, score in top5:
    print(score, filenames[idx])

# Show predicted image
Image.open(filenames[index_pos]).show()
