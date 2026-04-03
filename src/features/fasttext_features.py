import os
import argparse
import pandas as pd
import numpy as np
from gensim.models.fasttext import load_facebook_vectors
from tqdm import tqdm
from pathlib import Path 

# Default Config
ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = "phishguard_features.csv"
HERE = Path(__file__).resolve().parent
FASTTEXT_PATH = HERE/"fastText/cc.en.300.bin"
OUTPUT_DIR = "embeds_output"

class FastTextFeatureExtractor:
    """
    Loads the FastText model into memory ONCE. 
    Exposes methods for both batch training generation and single-item production inference.
    """
    def __init__(self, model_path: str = FASTTEXT_PATH):
        print(f"Loading FastText model from {model_path} (be patient)...")
        self.ft_model = load_facebook_vectors(model_path)
        self.embed_dim = self.ft_model.vector_size
        print(f"FastText loaded. Embedding dimension: {self.embed_dim}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        PRODUCTION MODE: Takes a single string and returns its vector embedding.
        """
        # Handle empty or invalid inputs safely
        if not text or not isinstance(text, str):
            return np.zeros(self.embed_dim, dtype=np.float32)

        words = text.split()
        vectors = []
        
        for w in words:
            if w in self.ft_model:
                vectors.append(self.ft_model[w])
                
        if not vectors:
            return np.zeros(self.embed_dim, dtype=np.float32)
            
        return np.mean(vectors, axis=0).astype(np.float32)

    def generate_training_data(self, features_csv: str, output_dir: str):
        """
        TRAINING MODE: Processes the entire CSV and outputs .npy matrices.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading features from {features_csv}...")
        df = pd.read_csv(features_csv)

        texts = df["clean_text"].fillna("").astype(str).values
        labels = df["label"].astype(np.int32).to_numpy()
        print(f"Loaded {len(texts)} samples.")

        print("Generating embeddings for entire dataset...")
        embedds = np.zeros((len(texts), self.embed_dim), dtype=np.float32)

        for i in tqdm(range(len(texts))):
            embedds[i] = self.get_embedding(texts[i])

        print("Embeddings generated.")

        x_path = os.path.join(output_dir, "X_embeddings.npy")
        y_path = os.path.join(output_dir, "y_labels.npy")
        
        np.save(x_path, embedds)
        np.save(y_path, labels)

        print(f"Saved successfully:\n -> {x_path}\n -> {y_path}")


# ==========================================
# Command Line Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastText Feature Extraction Module")
    parser.add_argument("mode", choices=["train", "prod"], help="Run mode: 'train' generates .npy files. 'prod' runs a quick test.")
    parser.add_argument("--csv", default=FEATURES_PATH, help="Path to input CSV (Train mode only)")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Output directory (Train mode only)")
    parser.add_argument("--model", default=FASTTEXT_PATH, help="Path to FastText .bin file")
    
    args = parser.parse_args()

    # Initialize the extractor
    extractor = FastTextFeatureExtractor(model_path=args.model)

    if args.mode == "train":
        print("--- RUNNING IN TRAINING MODE ---")
        extractor.generate_training_data(args.csv, args.out)
        
    elif args.mode == "prod":
        print("--- RUNNING IN PRODUCTION (TEST) MODE ---")
        test_text = "urgent password reset click link <URL>"
        vector = extractor.get_embedding(test_text)
        print(f"Test Input: '{test_text}'")
        print(f"Output Vector Shape: {vector.shape}")
        print(f"Output Vector Preview: {vector[:5]} ...")
