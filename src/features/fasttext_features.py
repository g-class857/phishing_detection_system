import os # for OS interface to creat directories, joining paths
import argparse # parse commandline arguments
import pandas as pd # data manipulation, read csv
import numpy as np # numerical operations creating arrays, handles vector averaging, saves .npy
from gensim.models.fasttext import load_facebook_vectors # loads pre-trained fastText model
from tqdm import tqdm # Displays progress bars during long loops 
from pathlib import Path # Object‑oriented filesystem paths – makes path manipulation cleaner and cross‑platform.
 
# Default Config
ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = "phishguard_features.csv"
HERE = Path(__file__).resolve().parent
FASTTEXT_PATH = HERE/"fastText/cc.en.300.bin"
OUTPUT_DIR = "embeds_output"

class FastTextFeatureExtractor:
    """
    Loads the FastText model into memory ONCE. memory efficient
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
        # Handle empty or invalid inputs safely. if so return zero vector
        if not text or not isinstance(text, str):
            return np.zeros(self.embed_dim, dtype=np.float32)

        words = text.split() # split the text to words
        vectors = []

        # retrieve the word vector if the word exists in FT vocabulary
        for w in words:
            if w in self.ft_model:
                vectors.append(self.ft_model[w]) # collect all word vectors to a list
                
        if not vectors:
            return np.zeros(self.embed_dim, dtype=np.float32) # return zero vector if not OOV 
            
        return np.mean(vectors, axis=0).astype(np.float32) # compute the element‑wise mean of all word vectors to obtain a single sentence embedding.
         # float32 to save memory and ensure compatibility with ML frameworks.
 
    def generate_training_data(self, features_csv: str, output_dir: str):
        """
        TRAINING MODE: Processes the entire CSV and outputs .npy matrices.
        """
        os.makedirs(output_dir, exist_ok=True) # creat directory if not exists
        
        print(f"Loading features from {features_csv}...")
        df = pd.read_csv(features_csv)

        # fillna("") replaces any missing (NaN) values with an empty string. astype(str) ensures all values are strings. .values returns a numpy array of the column.
        texts = df["clean_text"].fillna("").astype(str).values
        labels = df["label"].astype(np.int32).to_numpy() # converts the label column to 32‑bit integers and returns a numpy array.
        print(f"Loaded {len(texts)} samples.")

        print("Generating embeddings for entire dataset...")
        embedds = np.zeros((len(texts), self.embed_dim), dtype=np.float32) # pre‑allocates a 2D array of zeros to hold all embeddings. This is faster than appending.

        for i in tqdm(range(len(texts))): # show the progress bar while iterating over the text
            embedds[i] = self.get_embedding(texts[i])

        print("Embeddings generated.")

        x_path = os.path.join(output_dir, "X_embeddings.npy")  # saves the 2D array as a binary .npy file. efficient and preserves data types. 
        y_path = os.path.join(output_dir, "y_labels.npy")
        
        np.save(x_path, embedds)
        np.save(y_path, labels) # saves labels

        print(f"Saved successfully:\n -> {x_path}\n -> {y_path}")


# ==========================================
# Command Line Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastText Feature Extraction Module") # ArgumentParser creats parser object
    parser.add_argument("mode", choices=["train", "prod"], help="Run mode: 'train' generates .npy files. 'prod' runs a quick test.") # add_argument() defines arguments
    parser.add_argument("--csv", default=FEATURES_PATH, help="Path to input CSV (Train mode only)")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Output directory (Train mode only)")
    parser.add_argument("--model", default=FASTTEXT_PATH, help="Path to FastText .bin file")
    
    args = parser.parse_args() # read command with attributes named after the arguments

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
