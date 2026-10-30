# Bilingual Speech Emotion Recognition

## Run locally
1. Install Python 3.10 or 3.11.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`
5. Open the local URL shown in the terminal.

The first run downloads pretrained models and may take several minutes. A GPU is helpful but not required.

## Features
- Audio upload (WAV, MP3, M4A, OGG, FLAC)
- Microphone recording (supported by recent Streamlit versions/browsers)
- Multilingual Tamil/English speech transcription using Whisper
- Vocal emotion classification using a pretrained wav2vec2 emotion model
- English user interface and result labels

## Important limitation
The emotion classifier is trained on a limited speech-emotion benchmark and is not validated for Tamil. Tamil transcription is supported, but reliable Tamil vocal-emotion performance must be measured using a labelled Tamil speech dataset and native-speaker evaluation. Do not claim Tamil emotion accuracy without that validation. This is an academic prototype, not a clinical or high-stakes tool.
