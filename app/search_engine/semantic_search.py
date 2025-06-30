'''
This module will:
- Load audio features (e.g., from features.csv)
- Build a FAISS index from the numeric feature vectors
- Provide a method to:
  - Run semantic nearest neighbor search (e.g., “find 5 samples most similar to this one”)
  - Return the results with metadata (e.g., file names, tempo, etc.)

(Optional) Save/load the index for reuse
'''

import faiss
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    def __init__(
        self,
        features_path: str = "data/processed/features.csv",
        index_path: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
    ):
        self.features_path = Path(features_path)
        self.index_path = Path(index_path) if index_path else None
        self.exclude_cols = exclude_cols or ["file_name", "file_path", "estimated_key"]

        self.metadata_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None

        self._load_features()
        self._build_index()

    def _load_features(self):
        logger.info(f"Loading features from {self.features_path}")
        df = pd.read_csv(self.features_path)

        self.metadata_df = df.copy()
        feature_df = df.drop(columns=[col for col in self.exclude_cols if col in df.columns], errors="ignore")

        self.feature_matrix = feature_df.select_dtypes(include=[np.number]).values.astype("float32")
        logger.info(f"Loaded {self.feature_matrix.shape[0]} samples with {self.feature_matrix.shape[1]} features.")

    def _build_index(self):
        logger.info("Building FAISS index with L2 (Euclidean) distance...")
        d = self.feature_matrix.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.feature_matrix)

        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

        if self.index_path:
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Saved FAISS index to {self.index_path}")

    def query_by_vector(self, vector: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Query the index with a vector and return (index, distance)
        """
        vector = vector.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(vector, k)
        return list(zip(indices[0], distances[0]))

    def query_by_sample(self, file_name: str, k: int = 5) -> pd.DataFrame:
        """
        Run kNN query using the feature vector of a specific sample.
        """
        idx = self.metadata_df[self.metadata_df["file_name"] == file_name].index
        if len(idx) == 0:
            raise ValueError(f"Sample not found: {file_name}")

        query_vec = self.feature_matrix[idx[0]]
        results = self.query_by_vector(query_vec, k=k + 1)  # include itself

        results_df = self.metadata_df.iloc[[i for i, _ in results]]
        results_df["distance"] = [d for _, d in results]

        # Drop the query itself
        results_df = results_df[results_df["file_name"] != file_name].head(k)

        return results_df.reset_index(drop=True)


# CLI Example
if __name__ == "__main__":
    searcher = SemanticSearchEngine()
    query_file = "110384_JazzNEW-061210-9746506.wav.mp3"

    try:
        results = searcher.query_by_sample(query_file, k=3)
        print("Top 5 similar samples:")
        print(results[["file_name", "tempo_bpm", "distance"]])
    except ValueError as e:
        print(e)

'''
Example output:

Top 5 similar samples:
         file_name  tempo_bpm  distance
0  bright_pad.wav        89.1     2.43
1  mellow_chord.wav     88.7     2.55
2  airy_loop.wav        90.3     2.71
...

'''