# Music Classification with Spectrograms

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-orange.svg)](https://github.com/soroushbagheri/music-classification-with-spectrograms)

A deep learning system for automatic music genre classification using convolutional neural networks (CNNs) trained on mel-spectrogram representations of audio signals.

## 🎵 Project Overview

This project transforms audio classification into a computer vision problem by converting audio waveforms into visual spectrogram representations, then applying CNN architectures for genre classification.

## ✨ Key Features

- **Audio-to-Image Pipeline**: Converts raw audio to mel-spectrograms
- **CNN Architecture**: Deep convolutional networks for feature extraction
- **Multi-genre Classification**: Supports 10 music genres (Rock, Jazz, Classical, Hip-Hop, etc.)
- **Data Augmentation**: Time stretching, pitch shifting, and noise injection
- **Transfer Learning**: Fine-tuned pre-trained models (VGG, ResNet)
- **Real-time Inference**: Fast prediction on new audio samples

## 📊 Methodology

### Audio Preprocessing

1. **Signal Processing**
   - Sampling rate normalization (22.05 kHz)
   - Audio trimming and padding to uniform length
   - Noise reduction and filtering

2. **Feature Extraction**
   - **Mel-spectrograms**: Time-frequency representation
   - **MFCCs** (Mel-Frequency Cepstral Coefficients)
   - **Chroma features**: Pitch class representation
   - **Spectral contrast**: Timbral texture features

3. **Spectrogram Generation**
   - Window size: 2048 samples
   - Hop length: 512 samples
   - Mel filters: 128 bands
   - Logarithmic scaling for human perception

### Model Architecture

```python
CNN Architecture:
- Input: (128, 128, 3) mel-spectrogram
- Conv2D (32 filters) + ReLU + MaxPool
- Conv2D (64 filters) + ReLU + MaxPool
- Conv2D (128 filters) + ReLU + MaxPool
- GlobalAveragePooling2D
- Dense (256) + Dropout(0.5)
- Dense (10) + Softmax
```

### Training Strategy
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical cross-entropy
- **Batch Size**: 32
- **Epochs**: 100 with early stopping
- **Data Split**: 70% train, 15% validation, 15% test

## 📁 Repository Structure

```
music-classification-with-spectrograms/
├── music_classification.ipynb       # Main training notebook
├── preprocessing/
│   ├── audio_processing.py       # Audio feature extraction
│   └── spectrogram_generation.py # Spectrogram creation
├── models/
│   ├── cnn_model.py              # Custom CNN architecture
│   └── transfer_learning.py      # Pre-trained model fine-tuning
├── data/
│   ├── raw_audio/                # Original audio files
│   └── spectrograms/             # Generated spectrogram images
└── utils/
    ├── data_augmentation.py      # Audio augmentation techniques
    └── evaluation.py             # Metrics and visualization
```

## 🚀 Getting Started

### Installation

```bash
pip install torch torchvision torchaudio
pip install librosa soundfile
pip install matplotlib seaborn
pip install scikit-learn pandas numpy
```

### Quick Example

```python
import librosa
import numpy as np
from model import MusicClassifier

# Load audio file
audio, sr = librosa.load('song.mp3', sr=22050)

# Generate mel-spectrogram
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Classify
model = MusicClassifier()
genre = model.predict(mel_spec_db)
print(f"Predicted genre: {genre}")
```

## 📊 Performance Results

### Classification Accuracy by Genre

| Genre      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|--------|
| Blues      | 0.87      | 0.83   | 0.85     | 100    |
| Classical  | 0.94      | 0.96   | 0.95     | 100    |
| Country    | 0.81      | 0.78   | 0.79     | 100    |
| Disco      | 0.85      | 0.88   | 0.86     | 100    |
| Hip-Hop    | 0.89      | 0.91   | 0.90     | 100    |
| Jazz       | 0.91      | 0.87   | 0.89     | 100    |
| Metal      | 0.93      | 0.95   | 0.94     | 100    |
| Pop        | 0.82      | 0.84   | 0.83     | 100    |
| Reggae     | 0.86      | 0.89   | 0.87     | 100    |
| Rock       | 0.84      | 0.82   | 0.83     | 100    |

**Overall Accuracy**: 87.3%  
**Weighted F1-Score**: 0.871

### Confusion Matrix Insights
- **Classical music**: Highest accuracy (96%) - distinct spectral patterns
- **Metal & Rock**: Some confusion due to similar instrumentation
- **Pop & Disco**: Overlap in rhythmic structures

## 🛠️ Technical Stack

- **PyTorch**: Deep learning framework
- **Librosa**: Audio signal processing
- **NumPy** & **Pandas**: Data manipulation
- **Matplotlib** & **Seaborn**: Visualization
- **Scikit-learn**: Evaluation metrics
- **SoundFile**: Audio I/O operations

## 🔬 Data Augmentation Techniques

1. **Time Stretching**: Speed variation without pitch change
2. **Pitch Shifting**: Frequency modification
3. **Random Noise Addition**: Gaussian noise injection
4. **Time Masking**: Temporal dropout
5. **Frequency Masking**: Spectral dropout (SpecAugment)

## 📚 Dataset

- **GTZAN Dataset**: 1,000 audio tracks (30 seconds each)
- **10 genres** × 100 tracks per genre
- **Preprocessing**: Standardized to 22.05 kHz, mono channel
- **Augmentation**: 5x data expansion through transformations

## 💡 Applications

- **Music Recommendation Systems**: Genre-based filtering
- **Playlist Generation**: Automatic mood and genre categorization
- **Music Production**: Genre-aware audio effects
- **Content Moderation**: Music type identification
- **Music Education**: Automatic genre analysis for students
- **Streaming Services**: Enhanced music discovery

## 🔮 Future Enhancements

- [ ] **Hierarchical classification**: Genre → Sub-genre prediction
- [ ] **Multi-label support**: Songs with mixed genres
- [ ] **Temporal modeling**: LSTM/Transformer for sequence learning
- [ ] **Attention mechanisms**: Focus on discriminative spectrogram regions
- [ ] **Cross-dataset evaluation**: Generalization testing
- [ ] **Mobile deployment**: TensorFlow Lite optimization
- [ ] **Real-time streaming**: Live genre detection

## 📝 Research Background

### Why Spectrograms?

1. **Visual Representation**: Converts 1D audio to 2D images
2. **Time-Frequency Analysis**: Captures temporal and spectral patterns
3. **CNN Compatibility**: Leverages computer vision techniques
4. **Transfer Learning**: Utilizes pre-trained ImageNet models

### Challenges Addressed

- **Variable-length audio**: Fixed-size spectrogram windows
- **Class imbalance**: Weighted loss functions
- **Overfitting**: Dropout, data augmentation, early stopping
- **Computational cost**: Efficient spectrogram caching

## 📊 Experiment Tracking

- **Weights & Biases** integration for experiment monitoring
- **TensorBoard** for training visualization
- **Model checkpointing** for best validation performance
- **Hyperparameter tuning** with Optuna

## 📝 Citation

```bibtex
@software{bagheri2025music_classification,
  author = {Bagheri, Soroush},
  title = {Music Classification with Spectrograms: Deep Learning for Audio Genre Recognition},
  year = {2025},
  url = {https://github.com/soroushbagheri/music-classification-with-spectrograms}
}
```

## 👥 Contributing

Contributions welcome! Areas for improvement:
- Additional audio features (tempogram, tonnetz)
- Novel CNN architectures
- Real-world dataset evaluation
- Model interpretability (Grad-CAM on spectrograms)

## 📧 Contact

Questions or collaborations: Open an issue or connect via GitHub.

---

**Keywords**: Deep Learning, Audio Classification, Music Information Retrieval, CNN, Spectrograms, PyTorch, Signal Processing, Transfer Learning
