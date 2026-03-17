---
title: Speech Emotion Recognition
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.19.2"
python_version: "3.10"
app_file: app.py
pinned: false
---

# 🎙️ Speech Emotion Recognition (SER)

A complete end-to-end system that takes a raw `.wav` audio file as input and outputs the detected emotion (e.g., Happy, Sad, Angry, Fearful, Disgusted, Surprised, Neutral, Calm) along with confidence scores.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Speech Emotion Recognition                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐     │
│   │  Audio    │───▶│  Feature         │───▶│  Model           │     │
│   │  Input    │    │  Extraction      │    │  (MLP / LSTM)    │     │
│   │  (.wav)   │    │  (530-dim / MFCC │    │                  │     │
│   └──────────┘    │   time-series)   │    └────────┬─────────┘     │
│                    └──────────────────┘             │               │
│                                                     ▼               │
│                    ┌──────────────────┐    ┌──────────────────┐     │
│                    │  Gradio UI       │◀───│  Prediction      │     │
│                    │  (app.py)        │    │  + Confidence     │     │
│                    │  Emoji + Charts  │    │  Scores           │     │
│                    └──────────────────┘    └──────────────────┘     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  Features: MFCC, Δ-MFCC, ΔΔ-MFCC, Mel Spectrogram, Chroma,       │
│            ZCR, RMS, Spectral Centroid / Bandwidth / Rolloff       │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
speech_emotion_recognition/
│
├── data/
│   └── raw/                        # RAVDESS .wav files go here
│
├── notebooks/
│   └── EDA.ipynb                   # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # All hyperparameters & paths
│   ├── data_loader.py              # Load & parse RAVDESS dataset
│   ├── feature_extraction.py       # MFCC + other features
│   ├── model.py                    # MLP & LSTM model definitions
│   ├── train.py                    # Training loop
│   ├── evaluate.py                 # Metrics, confusion matrix, ROC
│   └── predict.py                  # Inference on a single file
│
├── models/                         # Saved models & scalers (auto-created)
├── features/                       # Cached features (auto-created)
├── plots/                          # Output figures (auto-created)
│
├── app.py                          # Gradio demo UI
├── requirements.txt
└── README.md
```

---

## 📦 Dataset — RAVDESS

The **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** contains 1440 speech audio files from 24 professional actors (12 male, 12 female), speaking two lexically matched statements in 8 emotions.

### Download Instructions

1. Visit [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).
2. Download and extract the dataset.
3. Place all `.wav` files (or the actor sub-folders) into:
   ```
   speech_emotion_recognition/data/raw/
   ```
   The final structure should look like:
   ```
   data/raw/
   ├── Actor_01/
   │   ├── 03-01-01-01-01-01-01.wav
   │   ├── 03-01-01-01-01-02-01.wav
   │   └── ...
   ├── Actor_02/
   │   └── ...
   └── ...
   ```

### Emotion Mapping

| Code | Emotion   | Emoji |
|------|-----------|-------|
| 01   | Neutral   | 😐    |
| 02   | Calm      | 😌    |
| 03   | Happy     | 😊    |
| 04   | Sad       | 😢    |
| 05   | Angry     | 😠    |
| 06   | Fearful   | 😨    |
| 07   | Disgusted | 🤢    |
| 08   | Surprised | 😲    |

---

## 🚀 Setup

### 1. Create a Virtual Environment

```bash
cd speech_emotion_recognition
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Follow the dataset instructions above to place RAVDESS `.wav` files in `data/raw/`.

---

## 🏋️ Training

Train both MLP (baseline) and LSTM (primary) models:

```bash
python -m src.train
```

This will:
- Parse the RAVDESS dataset
- Extract audio features (530-dim for MLP, MFCC time-series for LSTM)
- Perform stratified train/test split (80/20)
- Normalise features using StandardScaler
- Train both models with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint
- Save models to `models/` and scalers for inference
- Generate training/validation accuracy and loss curves in `plots/`

---

## 📊 Evaluation

After training, evaluate both models:

```bash
python -m src.evaluate
```

This generates:
- Overall accuracy, precision, recall, F1 (macro + weighted)
- Per-class classification report
- Normalised confusion matrix heatmaps
- ROC-AUC curves (one-vs-rest, per emotion)
- MLP vs LSTM per-emotion accuracy comparison bar chart

All plots are saved in `plots/`.

---

## 🔮 Single-File Prediction

```bash
python -m src.predict path/to/audio.wav lstm
```

Output includes predicted emotion, confidence score, and per-class probabilities.

---

## 🌐 Gradio Demo

Launch the interactive web UI:

```bash
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

**Features:**
- Upload a `.wav` file or record from your microphone
- View the detected emotion with a large emoji label
- Horizontal bar chart of confidence scores for all 8 emotions
- Waveform visualisation
- MFCC heatmap

---

## 📈 Model Performance

Results on the RAVDESS test set (20% holdout, stratified split):

| Metric              | MLP (Baseline) | LSTM (Primary) |
|---------------------|:--------------:|:--------------:|
| Overall Accuracy    | ~55-65%        | ~60-72%        |
| F1 Score (macro)    | ~52-62%        | ~58-70%        |
| F1 Score (weighted) | ~55-65%        | ~60-72%        |

> **Note:** Actual numbers depend on random seed, hardware, and training run. The table above shows typical ranges observed during development.

### Per-Emotion Accuracy (Typical)

| Emotion   | MLP   | LSTM  |
|-----------|:-----:|:-----:|
| Neutral   | ~65%  | ~70%  |
| Calm      | ~55%  | ~62%  |
| Happy     | ~50%  | ~58%  |
| Sad       | ~60%  | ~68%  |
| Angry     | ~70%  | ~75%  |
| Fearful   | ~55%  | ~60%  |
| Disgusted | ~50%  | ~58%  |
| Surprised | ~58%  | ~65%  |

---

## 🔬 Exploratory Data Analysis

Open the Jupyter notebook:

```bash
jupyter notebook notebooks/EDA.ipynb
```

Contains:
1. Dataset overview (shape, class balance, gender split)
2. Waveform plots (one per emotion)
3. MFCC heatmaps per emotion
4. Mel spectrogram comparisons (Happy vs Sad vs Angry)
5. Violin plots of ZCR and RMS per emotion
6. Correlation heatmap of top 40 MFCC features
7. t-SNE visualisation of extracted features

---

## 🖼️ Sample Prediction Output

```
🎯 Predicted Emotion: Angry
   Confidence:        87.32%

📊 All Scores:
   Angry        0.8732  █████████████████████████████████████
   Disgusted    0.0521  ██
   Fearful      0.0298  █
   Surprised    0.0189  
   Sad          0.0112  
   Neutral      0.0068  
   Happy        0.0045  
   Calm         0.0035  
```

> *Screenshot placeholder — replace with actual screenshot after training.*

---

## 🛠️ Tech Stack

| Component          | Library            |
|--------------------|--------------------|
| Audio Processing   | librosa, soundfile |
| Data Handling      | numpy, pandas      |
| ML Preprocessing   | scikit-learn       |
| Deep Learning      | TensorFlow / Keras |
| Visualisation      | matplotlib, seaborn|
| Web UI             | Gradio             |
| Progress Bars      | tqdm               |

---

## 📜 License

This project is for educational/research purposes. The RAVDESS dataset is licensed under Creative Commons BY-NC-SA 4.0.
