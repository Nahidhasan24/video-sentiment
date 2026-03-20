import os
import uuid
import subprocess
from flask import Flask, request, jsonify

import cv2
import numpy as np
import onnxruntime as ort
from transformers import pipeline
from faster_whisper import WhisperModel

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# Models
# -------------------------
print("Loading Whisper model...")
whisper_model = WhisperModel("base", compute_type="int8")

print("Loading multilingual sentiment model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

print("Loading FER+ ONNX model...")
emotion_model_path = "models/emotion-ferplus-8.onnx"
emotion_session = ort.InferenceSession(emotion_model_path)
input_name = emotion_session.get_inputs()[0].name  # automatically detect input name
EMOTIONS = ["neutral", "happy", "surprise", "sad", "anger", "disgust", "fear", "contempt"]

print("Models loaded ✅")

# -------------------------
# Helpers
# -------------------------
def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def speech_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path, language=None, task="transcribe")
    return " ".join([seg.text for seg in segments])

def analyze_sentiment(text):
    if not text.strip():
        return {"label": "neutral", "score": 0.5}
    return sentiment_model(text[:512])[0]

# -------------------------
# Visual Emotion
# -------------------------
def analyze_visual_emotion(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(frame_count // max_frames, 1)

    emotions_accum = {e: 0 for e in EMOTIONS}
    count = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: 
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (64, 64)).astype(np.float32) / 255.0
            face_input = np.expand_dims(face_resized, axis=0)  # batch
            face_input = np.expand_dims(face_input, axis=0)     # channel

            # Run ONNX FER+ model using automatic input name
            outputs = emotion_session.run(None, {input_name: face_input})
            raw_scores = outputs[0][0]
            probs = np.exp(raw_scores) / np.sum(np.exp(raw_scores))  # softmax

            for e, p in zip(EMOTIONS, probs):
                emotions_accum[e] += p
            count += 1

    cap.release()

    if count == 0:
        return {"status": "no_faces_detected"}

    avg_emotions = {e: float(emotions_accum[e]/count) for e in EMOTIONS}
    dominant = max(avg_emotions, key=avg_emotions.get)
    avg_emotions["dominant"] = dominant

    return avg_emotions

# -------------------------
# API endpoint
# -------------------------
@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]

    vid = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{vid}.mp4")
    audio_path = os.path.join(UPLOAD_FOLDER, f"{vid}.wav")

    file.save(video_path)

    # Audio -> text
    extract_audio(video_path, audio_path)
    text = speech_to_text(audio_path)

    # Sentiment
    sentiment = analyze_sentiment(text)

    # Visual emotion
    visual_emotion = analyze_visual_emotion(video_path)

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "visual_emotion": visual_emotion
    })

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)