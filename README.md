🎬 Bollywood Celebrity Face Recognition

📝 Project Description
This is a deep learning-based face recognition system that identifies Bollywood celebrities from images. The system uses FaceNet (a pre-trained deep neural network) to extract unique face embeddings and compares them with cosine similarity to find the closest match.

📂 Project Structure

bollywood_face_recognition/
│
├── bollywood_face/              # 📊 Dataset
│   ├── actor_1/
│   │   └── image.jpg
│   ├── actor_2/
│   │   └── image.jpg
│   └── ...
│
├── uploads/                      # 📁 Uploaded images folder
├── sample/                      # 📁 Sample test images
│
├── app.py                       # 🚀 Main Streamlit app
├── feature_extraction.py       # 🔧 Feature extraction script
├── test_facenet.py              # 🧪 Local testing script
│
├── embedding_facenet.pkl        # 💾 Pre-computed embeddings
├── filenames.pkl               # 💾 Image file paths
│
├── requirements.txt            # 📋 Python dependencies
└── README.md                   # 📖 Project readme


## ✨ Features

- ✅ Face detection using Haar Cascade
- ✅ FaceNet embedding extraction
- ✅ Cosine similarity-based matching
- ✅ Top-K voting for robust prediction
- ✅ Interactive Streamlit UI
- ✅ Real-time image upload and prediction


## 🧠 How It Works

* User uploads an image
* Image is preprocessed (resize + normalization)
* FaceNet extracts facial features (embeddings)
* Cosine similarity compares it with stored embeddings
* The most similar celebrity is displayed



---

## 🛠️ Tech Stack

* Python
* Streamlit
* Keras FaceNet
* NumPy
* Scikit-learn
* PIL (Image Processing)










