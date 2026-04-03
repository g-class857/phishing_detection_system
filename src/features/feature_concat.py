#!/usr/bin/env python3
"""
Feature pipeline for phishing detection.

Training mode (--mode train):
    - Align embeddings with dataset, extract manual features, scale, and save.

Production mode (Imported OR --mode prod):
    - Load schema + scaler into memory.
    - Accept embedding array + manual feature dict directly from other modules.
    - Produce model-ready vector.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from email.utils import parseaddr
from typing import Dict, Any, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feature_pipeline")

# ================= CONFIG =================
# Resolving absolute paths ensures it works no matter where the script is imported from
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed/embed_output"
MODEL_DIR = PROJECT_ROOT / "models"

CSV_PATH = DATA_DIR / "../phishguard_features.csv"
EMBED_PATH = DATA_DIR / "X_embeddings.npy"
LABEL_PATH = DATA_DIR / "y_labels.npy"

SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
SCALER_PATH = MODEL_DIR / "manual_scaler.pkl"
OUTPUT_PATH = DATA_DIR / "xgboost_features.npz"

# ================= FEATURE DEFINITIONS =================
NUMERIC_BASE = [
    "urgent_words_count",
    "digit_ratio",
    "body_entropy",
    "html_present",
    "auth_headers_present",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "received_count",
]

COUNT_FEATURES = [
    "urls_count",
    "domains_count",
    "ip_urls_count",
    "attachment_names_count",
]

HEADER_FEATURES = [
    "return_path_mismatch"
]

MANUAL_FEATURES = NUMERIC_BASE + COUNT_FEATURES + HEADER_FEATURES

# ================= HEADER UTILS =================
def extract_domain(email_string: str) -> str:
    if not isinstance(email_string, str) or not email_string:
        return ""
    _, addr = parseaddr(email_string)
    if "@" in addr:
        return addr.split("@")[-1].lower()
    return ""

def return_path_mismatch(row: pd.Series) -> int:
    # Use from_header, fallback to sender if needed
    sender = extract_domain(row.get("from_header", row.get("sender", "")))
    return_path = extract_domain(row.get("return_path", ""))

    if sender and return_path and sender != return_path:
        return 1
    return 0

# ================= PARALLEL FEATURE EXTRACTION =================
def extract_manual_features(df: pd.DataFrame) -> np.ndarray:
    logger.info("Vectorized feature extraction...")

    for col in NUMERIC_BASE + COUNT_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df[NUMERIC_BASE + COUNT_FEATURES] = df[NUMERIC_BASE + COUNT_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    # Vectorized return_path mismatch
    # Checking both 'from_header' and 'sender' to ensure robustness
    sender_col = df["from_header"] if "from_header" in df.columns else df.get("sender", pd.Series([""]*len(df)))
    sender_domains = sender_col.apply(extract_domain)
    return_domains = df.get("return_path", pd.Series([""]*len(df))).apply(extract_domain)

    df["return_path_mismatch"] = (
        (sender_domains != "") &
        (return_domains != "") &
        (sender_domains != return_domains)
    ).astype(int)

    return df[MANUAL_FEATURES].values.astype(np.float32)

# ================= TRAINING PIPELINE =================
def train_pipeline():
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    logger.info("Loading dataset...")
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

    logger.info("Loading labels...")
    y = np.load(LABEL_PATH).astype(int)

    logger.info("Loading embeddings...")
    embeddings = np.load(EMBED_PATH)

    # Align lengths
    min_len = min(len(df), len(embeddings), len(y))
    df = df.iloc[:min_len]
    embeddings = embeddings[:min_len]
    y = y[:min_len]
    
    logger.info("Removing unlabeled rows (-1)...")
    mask = y != -1
    df = df.loc[mask].reset_index(drop=True)
    embeddings = embeddings[mask]
    y = y[mask]

    logger.info(f"Remaining labeled samples: {len(y)}")
    if len(y) == 0:
        raise ValueError("No labeled samples found after filtering")
        
    logger.info("Extracting manual features...")
    X_manual = extract_manual_features(df)
    
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Label distribution: {dict(zip(unique, counts))}")
    
    logger.info("Scaling manual features...")
    scaler = StandardScaler()
    X_manual = scaler.fit_transform(X_manual)
    joblib.dump(scaler, SCALER_PATH)

    logger.info("Stacking embeddings...")
    X_emb = embeddings.astype(np.float32)

    logger.info("Concatenating features...")
    X = np.hstack([X_emb, X_manual]).astype(np.float32)

    logger.info("Saving training features...")
    np.savez_compressed(OUTPUT_PATH, X=X, y=y)

    schema = {
        "embedding_dim": X_emb.shape[1],
        "manual_features": MANUAL_FEATURES,
        "feature_order": [f"emb_{i}" for i in range(X_emb.shape[1])] + MANUAL_FEATURES
    }

    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=4)

    logger.info("Training pipeline completed.")
    logger.info(f"Feature matrix shape: {X.shape}")

# ================= PRODUCTION FEATURE BUILDER =================
class FeatureBuilder:
    """
    Import this class into your main pipeline module.
    It loads the schema/scaler ONCE upon initialization.
    """
    def __init__(self):
        if not SCHEMA_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError("Schema or Scaler not found. Run --mode train first.")

        with open(SCHEMA_PATH) as f:
            schema = json.load(f)

        self.embedding_dim = schema["embedding_dim"]
        self.manual_features = schema["manual_features"]
        self.scaler = joblib.load(SCALER_PATH)
        logger.info("FeatureBuilder initialized in Production Mode.")

    def build_vector(self, embedding_vector: np.ndarray, manual_dict: Dict[str, Any]) -> np.ndarray:
        """
        Takes the FastText array and the Preprocessed dictionary, 
        returns the finalized 1D Numpy array for XGBoost.
        """
        if len(embedding_vector) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding_vector)}")

        # Compute mismatch dynamically
        sender = manual_dict.get("from_header", manual_dict.get("sender", ""))
        ret_path = manual_dict.get("return_path", "")
        
        manual_dict["return_path_mismatch"] = (
            1 if extract_domain(sender) != extract_domain(ret_path) else 0
        )

        # Extract features in exact order as schema
        manual_values = []
        for feat in self.manual_features:
            val = float(manual_dict.get(feat, 0.0))
            manual_values.append(val)

        # Reshape for scaler (1 sample, n features)
        manual_array = np.array(manual_values).reshape(1, -1)
        scaled_manual = self.scaler.transform(manual_array)

        # Concatenate: [Embeddings, Scaled Manual Features]
        vector = np.hstack([embedding_vector.reshape(1, -1), scaled_manual])

        return vector.astype(np.float32)

# ================= CLI =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhishGuard Feature Pipeline")
    parser.add_argument("--mode", choices=["train", "prod"], required=True, 
                        help="train: builds scaler/schema. prod: runs a self-test.")
    args = parser.parse_args()

    if args.mode == "train":
        train_pipeline()

    elif args.mode == "prod":
        logger.info("Running Production Self-Test...")
        builder = FeatureBuilder()

        # Dummy data simulating what the Orchestrator will pass
        fake_embedding = np.random.rand(builder.embedding_dim)
        fake_manual = {
            "urgent_words_count": 2,
            "digit_ratio": 0.1,
            "body_entropy": 4.2,
            "html_present": 1,
            "auth_headers_present": 1,
            "spf_result": 1,
            "dkim_result": 1,
            "dmarc_result": 0,
            "received_count": 5,
            "urls_count": 2,
            "domains_count": 1,
            "ip_urls_count": 0,
            "attachment_names_count": 1,
            "sender": "attacker@evil.com",
            "return_path": "support@bank.com"
        }

        final_vector = builder.build_vector(fake_embedding, fake_manual)
        print("\nSUCCESS!\n ")
        print(f"Input Embedding Dim : {len(fake_embedding)}")
        print(f"Input Manual Feats  : {len(fake_manual)}")
        print(f"Output Vector Shape : {final_vector.shape}")

