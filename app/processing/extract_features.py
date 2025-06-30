'''
This module will:
- Load audio files from data/raw/audio/
- Extract a set of relevant audio features:
  - Tempo (BPM) and key estimation
  - Spectral features (centroid, rolloff, bandwidth)
  - MFCCs (Mel-frequency cepstral coefficients)
  - Embedding (e.g., using a pre-trained model like VGGish or OpenL3)
  - Save the feature vectors (likely in .json, .csv, or .npy)
'''

import os
import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    def __init__(self, audio_dir: str = "data/raw/audio", output_path: str = "data/processed/features.csv", sr: int = 22050):
        self.audio_dir = Path(audio_dir)
        self.output_path = Path(output_path)
        self.sr = sr  # Target sample rate

        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def extract_features_from_file(self, file_path: Path) -> Optional[Dict]:
        try:
            y, _ = librosa.load(file_path, sr=self.sr, mono=True)
            features = {}

            # Basic identifiers
            features["file_name"] = file_path.name
            features["file_path"] = str(file_path)

            # Duration
            features["duration_sec"] = librosa.get_duration(y=y, sr=self.sr)

            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
            features["tempo_bpm"] = tempo

            # Key estimation using chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
            key_index = chroma.mean(axis=1).argmax()
            features["estimated_key"] = librosa.hz_to_note(librosa.midi_to_hz(key_index + 24))  # crude approx

            # Spectral features
            features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sr))
            features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=self.sr))
            features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.sr))

            # MFCCs (first 13 coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
            for i in range(mfcc.shape[0]):
                features[f"mfcc_{i+1}"] = np.mean(mfcc[i])

            return features

        except Exception as e:
            logger.warning(f"Failed to extract features from {file_path.name}: {e}")
            return None

    def process_directory(self, limit: Optional[int] = None):
        logger.info(f"Processing audio files in: {self.audio_dir}")
        audio_files = list(self.audio_dir.glob("*.mp3")) + list(self.audio_dir.glob("*.wav"))
        if limit:
            audio_files = audio_files[:limit]

        feature_list: List[Dict] = []

        for file_path in audio_files:
            logger.info(f"Extracting features from: {file_path.name}")
            feats = self.extract_features_from_file(file_path)
            if feats:
                feature_list.append(feats)

        if feature_list:
            df = pd.DataFrame(feature_list)
            df.to_csv(self.output_path, index=False)
            logger.info(f"Saved extracted features to: {self.output_path}")
        else:
            logger.warning("No features extracted. Nothing saved.")


# CLI Usage
if __name__ == "__main__":
    extractor = AudioFeatureExtractor()
    extractor.process_directory()

'''
Example output:

| file\_name      | tempo\_bpm | estimated\_key | spectral\_centroid | mfcc\_1 | ... |
| --------------- | ---------- | -------------- | ------------------ | ------- | --- |
| 103298\_pad.mp3 | 88.2       | C#             | 1234.88            | -250.2  | ... |

'''