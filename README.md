# Music Genre Classification with Spectrograms

Deep learning-based music genre classification using spectrogram representations and convolutional neural networks.

## 🎵 Overview

This project implements an audio classification system that converts music signals into visual spectrograms and uses CNN architectures to identify music genres. The approach leverages the time-frequency representation of audio to capture musical patterns and textures.

## 🎯 Objectives

- **Audio Processing**: Convert raw audio to mel-spectrograms for visual representation
- **Deep Learning**: Train CNN models on spectrogram images for genre classification
- **Feature Engineering**: Extract meaningful audio features from time-frequency domains
- **Model Evaluation**: Compare different architectures and hyperparameters

## 📊 Dataset

- **Source**: Music genre dataset with labeled audio files
- **Preprocessing**: Audio normalization, resampling, and segmentation
- **Features**: Mel-spectrograms, MFCCs, chroma features
- **Classes**: Multiple music genres (rock, classical, jazz, pop, etc.)

## 🔬 Methodology

### 1. Audio to Spectrogram Conversion
```python
import librosa
import librosa.display

# Load audio and create mel-spectrogram
y, sr = librosa.load(audio_file)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
```

### 2. CNN Architecture
- **Input**: Spectrogram images (time × frequency)
- **Layers**: Convolutional layers for feature extraction
- **Pooling**: Max pooling for dimensionality reduction
- **Output**: Genre classification (softmax)

### 3. Training Pipeline
- Data augmentation (time stretching, pitch shifting)
- Transfer learning from pre-trained vision models
- Cross-validation for robust evaluation

## 🛠️ Technical Stack

- **Audio Processing**: librosa, soundfile
- **Deep Learning**: PyTorch / TensorFlow + Keras
- **Visualization**: matplotlib, librosa.display
- **Data Handling**: numpy, pandas
- **Environment**: Jupyter Notebook

## 📦 Installation

```bash
pip install librosa soundfile
pip install torch torchvision  # or tensorflow
pip install matplotlib numpy pandas scikit-learn
```

## 🚀 Usage

```python
# Load and preprocess audio
import librosa
audio, sr = librosa.load('music.wav', sr=22050)

# Create spectrogram
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
log_mel_spec = librosa.power_to_db(mel_spec)

# Predict genre
genre = model.predict(log_mel_spec)
print(f"Predicted Genre: {genre}")
```

## 📈 Results

- **Training Accuracy**: Achieved through iterative optimization
- **Validation Performance**: Cross-validated results
- **Confusion Matrix**: Genre-wise classification accuracy
- **Feature Importance**: Analysis of spectral patterns

## 🔑 Key Features

- **Mel-Spectrogram Analysis**: Time-frequency representation
- **Data Augmentation**: Robust training through audio transformations
- **Visualization**: Spectrogram plots and model interpretability
- **Modular Design**: Easy to extend to other audio tasks

## 📚 Applications

- Music recommendation systems
- Audio content moderation
- Music library organization
- Automatic playlist generation
- Music production tools

## 🔗 References

- **librosa**: Audio and music signal analysis library
- **Deep Learning for Audio**: Humphrey et al. - "Learnable Front-ends for Audio Classification"
- **Spectrogram CNNs**: Piczak - "Environmental sound classification with convolutional neural networks"

## 📄 Citation

```bibtex
@software{bagheri2025music_classification,
  author = {Bagheri, Soroush},
  title = {Music Genre Classification with Spectrograms},
  year = {2025},
  url = {https://github.com/soroushbagheri/music-classification-with-spectrograms}
}
```

## 🎓 Project Context

Developed as part of machine learning coursework focusing on audio processing and deep learning applications in music information retrieval (MIR).

## 📫 Contact

For questions or collaboration, please open an issue on GitHub.

---

**Keywords**: Audio Classification, Music Information Retrieval, CNN, Deep Learning, Spectrograms, Mel-spectrograms, PyTorch, Machine Learning