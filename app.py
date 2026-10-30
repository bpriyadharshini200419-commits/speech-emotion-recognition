import streamlit as st
import numpy as np
import librosa
import torch
from transformers import pipeline
from faster_whisper import WhisperModel
import tempfile, os

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

st.title("🎙️ Bilingual Speech Emotion Recognition")
st.caption("Tamil and English speech input • English interface and results")

st.info(
    "Record your voice or upload an audio file. The system analyses vocal emotion "
    "and transcribes Tamil/English speech. Emotion predictions are experimental "
    "and should not be used for medical, hiring, disciplinary, or other high-stakes decisions."
)

@st.cache_resource
def load_emotion_model():
    # Pretrained speech-emotion model. Its published training domain is not a guarantee
    # of equivalent performance on Tamil speech or every microphone/environment.
    return pipeline(
        "audio-classification",
        model="superb/wav2vec2-base-superb-er",
        device=0 if torch.cuda.is_available() else -1
    )

@st.cache_resource
def load_asr_model():
    # Multilingual Whisper model supports Tamil and English transcription.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel("small", device=device, compute_type=compute_type)

def save_uploaded_audio(uploaded):
    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(uploaded.getbuffer())
    f.close()
    return f.name

def normalize_emotion(label):
    label = label.lower()
    mapping = {
        "hap": "Happy", "happy": "Happy",
        "sad": "Sad", "ang": "Angry", "anger": "Angry",
        "neu": "Neutral", "neutral": "Neutral",
        "fear": "Fearful", "fea": "Fearful",
        "dis": "Disgust", "sur": "Surprised"
    }
    for key, value in mapping.items():
        if key in label:
            return value
    return label.replace("_", " ").title()

def analyse_audio(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    if len(audio) < sr:
        raise ValueError("Please provide at least one second of clear speech.")
    emotion_model = load_emotion_model()
    predictions = emotion_model({"raw": audio, "sampling_rate": sr}, top_k=None)
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    best = predictions[0]
    # Whisper detects the spoken language automatically; output is kept in original script.
    asr = load_asr_model()
    segments, info = asr.transcribe(path, beam_size=5)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    return normalize_emotion(best["label"]), float(best["score"]), predictions, transcript, info.language

audio_file = st.file_uploader(
    "Upload speech audio", type=["wav", "mp3", "m4a", "ogg", "flac"]
)
recording = st.audio_input("Or record your voice")

selected = recording if recording is not None else audio_file

if selected is not None:
    st.audio(selected)
    if st.button("Analyse Speech", type="primary", use_container_width=True):
        path = save_uploaded_audio(selected)
        try:
            with st.spinner("Analysing speech. The first run may download AI models..."):
                emotion, confidence, scores, transcript, language = analyse_audio(path)
            st.subheader("Analysis Result")
            st.metric("Predicted vocal emotion", emotion.upper(), f"{confidence*100:.1f}% model score")
            st.write(f"**Detected speech language:** {language.upper()}")
            st.write("**Speech transcript:**")
            st.write(transcript if transcript else "No clear speech transcription was detected.")
            st.caption(
                "The score is the model's confidence-like output, not a calibrated probability. "
                "The vocal model has a limited emotion label set; Tamil performance needs local validation."
            )
            st.subheader("Model scores")
            for item in sorted(scores, key=lambda x: x["score"], reverse=True):
                st.write(f"{normalize_emotion(item['label'])}: {item['score']*100:.1f}%")
                st.progress(min(max(float(item["score"]), 0.0), 1.0))
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.write("Check the audio format, internet connection (for first-time model download), and installed packages.")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

with st.expander("About this project"):
    st.write(
        "This prototype uses a pretrained speech-emotion classifier for vocal cues and "
        "multilingual Whisper for Tamil/English transcription. The displayed emotion is "
        "based on the voice model, not simply on the meaning of the words. Speech emotion "
        "is subjective and can be affected by accent, background noise, speaking style, "
        "and recording quality."
    )
