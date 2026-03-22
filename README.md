# Video Sentiment Analysis API's

This Flask-based application analyzes the **sentiment** and **visual emotions** from videos. It supports **multiple languages** using Whisper and XLM-RoBERTa, and facial emotion detection with the **FER+ ONNX model**.

---

## Features

- **Speech-to-text** from video using [Whisper](https://github.com/openai/whisper)
- **Multilingual sentiment analysis** using [XLM-RoBERTa](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
- **Facial emotion recognition** using [FER+ ONNX model](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus)
- Supports **multiple languages** for audio transcription
- Returns **JSON output** with text, sentiment, and visual emotion probabilities

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Nahidhasan24/video-sentiment.git
cd video-sentiment
```

Create a Python virtual environment:

```bash
python -m venv venv
source venv/Scripts/activate    # Windows
# OR
source venv/bin/activate        # Linux/Mac
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Requirements file** should include:
>
> ```text
> flask
> onnxruntime
> opencv-python
> transformers
> torch
> torchvision
> torchaudio
> faster-whisper
> ```

---

## Setup ONNX Model

Download the FER+ ONNX model:

```bash
mkdir -p models
curl -L https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx -o models/emotion-ferplus-8.onnx
```

Make sure the path is:

```
video-sentiment/models/emotion-ferplus-8.onnx
```

---

## Running the API

Start the Flask server:

```bash
python main.py
```

The API will run on:

```
http://127.0.0.1:5000
```

---

## API Endpoint

### `POST /analyze`

**Request:**

- Form-data:
  - `video`: MP4 video file

**Example (using `curl`):**

```bash
curl -X POST -F "video=@test_video.mp4" http://127.0.0.1:5000/analyze
```

**Response:**

```json
{
  "text": "আমি আজ খুব খুশি",
  "sentiment": { "label": "POSITIVE", "score": 0.97 },
  "visual_emotion": {
    "neutral": 0.05,
    "happy": 0.75,
    "surprise": 0.1,
    "sad": 0.03,
    "anger": 0.03,
    "disgust": 0.01,
    "fear": 0.01,
    "contempt": 0.01,
    "dominant": "happy"
  }
}
```

---

## Notes

- **FER+ ONNX model input:** The server automatically detects the input tensor name.
- **Multilingual support:** Whisper handles multiple languages for speech recognition.
- **Visual emotion:** Average over sampled video frames; returns dominant emotion.

---

## Troubleshooting

- `onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile` → Make sure `emotion-ferplus-8.onnx` exists in `models/`.
- `ValueError: Required inputs [...] are missing` → This happens if you use a different ONNX model; the code auto-detects the input name.
- Ensure all dependencies are installed in the same Python environment.

---

## License

MIT License

---

## References

- [Whisper by OpenAI](https://github.com/openai/whisper)
- [XLM-RoBERTa Sentiment Model](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
- [FER+ ONNX Model](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus)

```

```
