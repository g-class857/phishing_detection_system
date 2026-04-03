import logging 
import json 
import hashlib
import time 
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple 

import numpy as np
import xgboost as xgb
import joblib

import sys  
import argparse  
import requests

try:
	import shap
	SHAP_AVAILABLE = True
except ImportError:
	SHAP_AVAILABLE = False

# Ensure these .py files are in the same directory or Python path

# ------------------------------------------------------------
# 1. Project Paths (GLOBAL SCOPE)
# ------------------------------------------------------------
# This finds the 'phishing detection system' folder
ROOT = Path(__file__).resolve().parent.parent 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)

# Define all file paths used by the Predictor
MODEL_PATH = MODEL_DIR / "phishguard_xgb.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
SCALER_PATH = MODEL_DIR / "manual_scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SHAP_BACKGROUND_PATH = MODEL_DIR / "shap_background.npy"
WAZUH_JSONL_PATH = LOG_DIR / "phishguard_predictions.jsonl"


try:
    from src.features.preprocess import production_preprocessing
    from src.features.fasttext_features import FastTextFeatureExtractor
    from src.features.feature_concat import FeatureBuilder
    from virus_total.VT_Client import VT_Client
except ImportError as e:
    print(f"CRITICAL: Missing custom module: {e}")
    sys.exit(1)
# ------------------------------------------------------------
# 2. Thresholds & Logging
# ------------------------------------------------------------
SAFE_THRESHOLD = 0.30
PHISHING_THRESHOLD = 0.70

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("PhishGuard.Predictor")

class PhishGuardPredictor:
    def _load_feature_names(self) -> List[str]:
        """
        Loads the exact order of features the model was trained on.
        This is critical for SHAP to display the correct labels.
        """
        if not SCHEMA_PATH.exists():
            logger.error(f"Feature schema missing at {SCHEMA_PATH}")
            # Fallback: if schema is missing, you might need to hardcode 
            # or raise an error depending on how strict your pipeline is.
            raise FileNotFoundError(f"Missing {SCHEMA_PATH}. SHAP cannot map features.")
            
        with open(SCHEMA_PATH, "r") as f:
            schema = json.load(f)
            
        # We assume your JSON has a key called 'feature_order'
        feature_names = schema.get("feature_order", [])
        logger.info(f"Loaded {len(feature_names)} feature definitions from schema.")
        return feature_names
        
    def __init__(self, vt_api_key: Optional[str] = None):
        logger.info("Initializing PhishGuard Production Engine...")
        
        # 1. Load XGBoost Model (JSON format)
        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_PATH))
        
        # 2. Load Feature Builder
        self.builder = FeatureBuilder()
        
        # 3. Load FastText (This loads the .bin file into RAM ONCE)
        self.ft_extractor = FastTextFeatureExtractor()
        
        # 4. Initialize VT Client
        # If vt_api_key is passed here, it overrides whatever is inside vt_client.py
        self.vt_client = VT_Client(api_key=vt_api_key) 

        # 5. Load SHAP
        self.feature_names = self._load_feature_names()
        self.explainer = self._init_shap()

    def _init_shap(self):
        if SHAP_AVAILABLE and SHAP_BACKGROUND_PATH.exists():
            try:
                bg_data = np.load(SHAP_BACKGROUND_PATH)
                return shap.TreeExplainer(self.model, data=bg_data)
            except Exception as e:
                logger.error(f"Failed to load SHAP explainer: {e}")
        return None

    def explain(self, vector: np.ndarray) -> List[Dict]:
        if not self.explainer:
            return []
        
        try:
            shap_values = self.explainer.shap_values(vector)
            if isinstance(shap_values, list): 
                shap_values = shap_values[0]
            
            contributions = []
            # Flatten vector for iteration
            flat_vector = vector.flatten() if hasattr(vector, "flatten") else vector[0]
            
            for i, val in enumerate(shap_values[0]):
                contributions.append({
                    "feature": self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}",
                    "impact": round(float(val), 4),
                    "value": float(flat_vector[i]),
                    "direction": "increases_risk" if val > 0 else "decreases_risk"
                })
            
            # Sort by absolute impact to find the top 5 reasons
            return sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)[:5]
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return []

    def predict(self, raw_email: str) -> Dict[str, Any]:
        start_ts = time.time()
        event_id = hashlib.sha256(raw_email.encode()).hexdigest()

        try:
            # 1. Preprocessing
            processed = production_preprocessing(raw_email)
            if not processed:
                return {"event_id": event_id, "status": "rejected", "reason": "invalid_format"}

            # 2. FastText Embedding
            clean_text = processed.get("clean_text", "")
            # Update this line if your FastText extractor method has a different name
            embedding = self.ft_extractor.get_embedding(clean_text)

            # 3. Build Final Vector 
            final_vector = self.builder.build_vector(embedding, processed)

            # 4. XGBoost Inference
            dmat = xgb.DMatrix(final_vector)
            prob = float(self.model.predict(dmat)[0])
            
            # 5. Threshold Logic
            if prob < SAFE_THRESHOLD:
                decision = "safe"
            elif prob < PHISHING_THRESHOLD:
                decision = "suspicious"
            else:
                decision = "phishing"

            # 6. Reputation (Enrichment via VT_Client)
            reputation = self.vt_client.get_reputations(
                urls=processed.get("urls", []),
                domains=processed.get("domains", []),
                ips=processed.get("ips", [])
            )

            # 7. SHAP Explanation
            reasons = self.explain(final_vector) if decision != "safe" else []

            # 8. Construct Output
            result = {
                "event_id": event_id,
                "timestamp": time.time(),
                "probability": round(prob, 4),
                "decision": decision,
                "reputation_report": reputation,
                "top_reasons": reasons,
                "metadata": {
                    "runtime_sec": round(time.time() - start_ts, 3),
                    "model_version": "1.0.0"
                }
            }

            # 9. Write to JSONL for Wazuh
            with open(WAZUH_JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")

            return result

        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}")
            return {"event_id": event_id, "status": "error", "error_message": str(e)}
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="PhishGuard Predictor CLI")
    parser.add_argument("--vt-key", help="Your VirusTotal API Key (Overrides default)")
    
    args, unknown = parser.parse_known_args()
    print("\n \t You wanna bypass me, huh, really haha ha, let's see what u have: \n \n \t press ctrl+d when u finish")

    # Read the raw email from stdin
    raw_email_input = sys.stdin.read()

    if not raw_email_input.strip():
        print("Error: No email data provided in stdin.")
        sys.exit(1)

    # Pass the user-provided API key (if any) to the Predictor
    predictor = PhishGuardPredictor(vt_api_key=args.vt_key)
    
    output = predictor.predict(raw_email_input)
    print(output)
    print(json.dumps(output, indent=2, ensure_ascii=False))
