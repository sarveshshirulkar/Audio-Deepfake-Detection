import os
import tempfile
from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

MODEL_PKL = os.path.join("model", "wav2vec2_audio_deepfake.pkl")
DEVICE = "cpu"  
SAMPLE_RATE = 16000          
CLIP_SECONDS = 5              
THRESHOLD = 0.5               

app = Flask(__name__)

class Wav2Vec2ForAudioDeepfake(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", num_labels=2, freeze_feature_extractor=True):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        if freeze_feature_extractor:
            try:
                for p in self.wav2vec2.feature_extractor.parameters():
                    p.requires_grad = False
            except Exception:
                pass
        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_values):
        # input_values: (B, seq_len)
        outputs = self.wav2vec2(input_values)           # BaseModelOutput
        last_hidden = outputs.last_hidden_state         # (B, T, H)
        pooled = last_hidden.mean(dim=1)                # (B, H)
        logits = self.classifier(pooled)                # (B, num_labels)
        return {"logits": logits}

# ----- load saved pkl bundle -----
def load_model_and_extractor(pkl_path, device="cpu"):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Model bundle not found at {pkl_path}")
    bundle = torch.load(pkl_path, map_location="cpu")
    model_name = bundle.get("model_name", "facebook/wav2vec2-base")
    state = bundle.get("model_state_dict", None)
    label_mapping = bundle.get("label_mapping", {0: "Real", 1: "Fake"})
    if state is None:
        # support legacy: maybe user saved whole model object
        # try to handle that
        try:
            model_obj = bundle.get("model", None)
            if model_obj is not None:
                model = model_obj.to(device)
                model.eval()
            else:
                raise RuntimeError("No model_state_dict found in pkl.")
        except Exception as e:
            raise RuntimeError("Could not find a state dict or model object in the pkl.") from e
    else:
        model = Wav2Vec2ForAudioDeepfake(model_name=model_name, num_labels=2, freeze_feature_extractor=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    return model, feature_extractor, label_mapping

# Load on startup
try:
    model, feature_extractor, label_mapping = load_model_and_extractor(MODEL_PKL, DEVICE)
    print("Model and feature extractor loaded. Label mapping:", label_mapping)
except Exception as e:
    model = None
    feature_extractor = None
    label_mapping = {0: "Real", 1: "Fake"}
    print("Warning: model failed to load on startup:", e)

# ----- helper: preprocess to match training -----
def load_and_prepare_audio(file_path, sr=SAMPLE_RATE, clip_seconds=CLIP_SECONDS):
    """
    Loads file, resamples to sr, converts to mono, pads/truncates to clip_seconds.
    Returns 1D numpy float32 array.
    """
    wav, orig_sr = librosa.load(file_path, sr=None, mono=True)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr, sr)
    target_len = int(sr * clip_seconds)
    if len(wav) > target_len:
        wav = wav[:target_len]
    elif len(wav) < target_len:
        pad = target_len - len(wav)
        wav = np.pad(wav, (0, pad), mode='constant')
    return wav.astype(np.float32)

# ----- inference helper -----
def predict_from_numpy(wav_np):
    """
    wav_np: 1-D numpy array sampled at SAMPLE_RATE and length CLIP_SECONDS * SAMPLE_RATE
    Returns {label, confidence}
    """
    if model is None or feature_extractor is None:
        return {"error": "Model not loaded on server."}

    # feature extractor -> returns dict with 'input_values'
    inputs = feature_extractor(wav_np, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)  # shape (1, seq_len)
    with torch.no_grad():
        out = model(input_values)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()  # [p_real, p_fake] per label_mapping used in notebook
    # Notebook saved label_mapping {0:"Real", 1:"Fake"}
    # We assume index 1 == Fake
    # If label_mapping uses different order, adapt below:
    # find which index corresponds to "Fake" (case-insensitive)
    fake_index = None
    for idx, lbl in label_mapping.items():
        if str(lbl).lower() == "fake":
            fake_index = int(idx)
    if fake_index is None:
        # fallback: assume index 1 is fake
        fake_index = 1 if len(probs) > 1 else 0

    prob_fake = float(probs[fake_index])
    label = "fake" if prob_fake >= THRESHOLD else "real"
    return {"label": label, "confidence": round(prob_fake, 4)}

# ----- routes -----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "no selected file"}), 400

    # Save upload to temp file
    suffix = os.path.splitext(f.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_name = tmp.name
        f.save(tmp_name)

    try:
        # Preprocess -> numpy waveform matching training
        wav_np = load_and_prepare_audio(tmp_name, sr=SAMPLE_RATE, clip_seconds=CLIP_SECONDS)
        result = predict_from_numpy(wav_np)
        os.unlink(tmp_name)
        return jsonify(result)
    except Exception as e:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
