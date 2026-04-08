import logging 
import json 
import hashlib
import time 
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple 

import numpy as np
import xgboost as xgb
import joblib

import re
import urllib.parse 
import difflib 

import sys  
import argparse  
import requests

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# --- Project Paths ---
ROOT = Path(__file__).resolve().parent.parent 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "phishguard_xgb.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
WAZUH_JSONL_PATH = LOG_DIR / "phishguard_predictions.jsonl"
SHAP_BACKGROUND_PATH = MODEL_DIR / "shap_background.npy"

try:
    from src.features.preprocess import production_preprocessing
    from src.features.fasttext_features import FastTextFeatureExtractor
    from src.features.feature_concat import FeatureBuilder
    from virus_total.VT_Client import VT_Client
except ImportError as e:
    print(f"CRITICAL: Missing custom module: {e}")
    sys.exit(1)

SAFE_THRESHOLD = 0.30
PHISHING_THRESHOLD = 0.70
TARGET_BRANDS = ["microsoft", "apple", "paypal", "amazon", "google", "netflix", "facebook"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("PhishGuard.Predictor")

def transliterate_homoglyphs(text: str) -> str:
    """Translates common visual tricks back to standard ASCII."""
    cmap = {
        'а': 'a', 'с': 'c', 'е': 'e', 'о': 'o', 'р': 'p', 'х': 'x', 'у': 'y', # Cyrillic
        'і': 'i', 'ј': 'j', 'ѕ': 's', 'ԁ': 'd', 'ԛ': 'q', 'ԝ': 'w',
        'α': 'a', 'ο': 'o', 'ν': 'v', 'ρ': 'p', 'τ': 't', 'μ': 'u',           # Greek
        '0': 'o', '1': 'l', '3': 'e', '5': 's'                                # Numeric
    }
    # Handle the classic 'rn' masquerading as 'm'
    text = text.replace('rn', 'm')
    return ''.join(cmap.get(c, c) for c in text)
    
class PhishGuardPredictor:
    def __init__(self, vt_api_key: Optional[str] = None):
        logger.info("Initializing PhishGuard Production Engine...")
        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_PATH))
        self.builder = FeatureBuilder()
        self.ft_extractor = FastTextFeatureExtractor()
        self.vt_client = VT_Client(api_key=vt_api_key) 
        self.feature_names = self._load_feature_names()
        self.explainer = self._init_shap()

    def _load_feature_names(self) -> List[str]:
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Missing {SCHEMA_PATH}. SHAP cannot map features.")
        with open(SCHEMA_PATH, "r") as f:
            return json.load(f).get("feature_order", [])

    def _init_shap(self):
        if SHAP_AVAILABLE and SHAP_BACKGROUND_PATH.exists():
            try:
                bg_data = np.load(SHAP_BACKGROUND_PATH)
                return shap.TreeExplainer(self.model, data=bg_data)
            except Exception as e:
                logger.error(f"Failed to load SHAP: {e}")
        return None
        
    def _evaluate_security_heuristics(self, processed_dict: Dict[str, Any]) -> List[Dict]:
        alerts = []
        TARGET_BRANDS = ["apple", "amazon", "paypal", "microsoft", "google"]
        raw_text = processed_dict.get("raw_text", "")
        
        # --- 1. AGGRESSIVE DATA EXTRACTION ---
        domains_to_check = set()
        
        # A. Extract Sender from Raw Text
        sender_domain = ""
        from_match = re.search(r'^From:\s*.*<(.+?)>', raw_text, re.M | re.I)
        if from_match:
            sender_domain = from_match.group(1).split('@')[-1].strip("<>\"' ").lower()
            domains_to_check.add(sender_domain)

        # B. Extract EVERY URL from Raw Text (handles hidden links and plain text)
        all_urls = re.findall(r'https?://[^\s<>"\'()]+', raw_text, re.I)
        for link in all_urls:
            try:
                netloc = urllib.parse.urlparse(link).netloc.lower().split(':')[0]
                if netloc:
                    domains_to_check.add(netloc)
            except:
                continue

        # --- 2. DOMAIN ANALYSIS ---
        for raw_domain in domains_to_check:
            # Decode Punycode if present
            try:
                unicode_domain = raw_domain.encode('utf-8').decode('idna')
            except:
                unicode_domain = raw_domain
            
            domain_core = unicode_domain.split('.')[0].lower()
            core_no_hyphens = domain_core.replace('-', '')
            
            # Translate homoglyphs to see what word they are trying to hide
            normalized_core = transliterate_homoglyphs(core_no_hyphens)
            has_tricks = (normalized_core != core_no_hyphens)

            # Test against target brands
            for brand in TARGET_BRANDS:
                
                # TEST A: Substring Impersonation (Catches "pay-pal-security")
                # If 'paypal' is in 'paypalsecurity', but it isn't exactly 'paypal.com'
                if brand in normalized_core and normalized_core != brand:
                    alerts.append({
                        "feature": f"Brand Impersonation ({brand})",
                        "impact_score": 999.8,
                        "actual_value": raw_domain,
                        "direction": "increases_risk"
                    })
                    break # Prevent duplicate brand alerts
                    
                # TEST B: Exact Homoglyph Match (Catches "аррle")
                # If translating tricks results in an exact brand match
                if has_tricks and normalized_core == brand:
                    alerts.append({
                        "feature": f"Homoglyph Deception ({brand})",
                        "impact_score": 999.7,
                        "actual_value": unicode_domain,
                        "direction": "increases_risk"
                    })
                    break

                # TEST C: Standard Typosquatting (Catches "amzaon")
                for part in domain_core.split('-'):
                    norm_part = transliterate_homoglyphs(part)
                    sim = difflib.SequenceMatcher(None, brand, norm_part).ratio()
                    if 0.80 <= sim < 1.0: # Bumped to 0.80 to avoid short-word false positives
                        alerts.append({
                            "feature": f"Levenshtein: Brand Spoof ({brand})",
                            "impact_score": 999.8,
                            "actual_value": f"Matched '{part}' ({round(sim*100)}%)",
                            "direction": "increases_risk"
                        })
                        break

            # TEST D: IP Address Host (Catches "192.168.4.12")
            if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', raw_domain):
                alerts.append({
                    "feature": "Critical: IP Address as Host",
                    "impact_score": 999.9,
                    "actual_value": raw_domain,
                    "direction": "increases_risk"
                })

        # --- 3. SENDER SPOOFING (Return-Path Check) ---
        rp_match = re.search(r'^Return-Path:\s*<(.+?)>', raw_text, re.M | re.I)
        if rp_match and sender_domain:
            rp_domain = rp_match.group(1).split('@')[-1].strip("<>\"' ").lower()
            if rp_domain != sender_domain:
                alerts.append({
                    "feature": "Critical: Sender Spoofing",
                    "impact_score": 999.6,
                    "actual_value": f"From: {sender_domain} vs Return: {rp_domain}",
                    "direction": "increases_risk"
                })

        return alerts
    def explain(self, vector: np.ndarray, raw_dict: Dict[str, Any], decision: str) -> List[Dict]:
        if not self.explainer: return []
        try:
            shap_values = self.explainer.shap_values(vector)
            if isinstance(shap_values, list): shap_values = shap_values[0]
            
            contributions = []
            for i, val in enumerate(shap_values[0]):
                feature_name = self.feature_names[i]
                if feature_name.startswith("emb_") or val == 0: continue
                
                contributions.append({
                    "feature": feature_name.replace("_", " ").title(),
                    "impact_score": round(float(val), 4),
                    "direction": "increases_risk" if val > 0 else "decreases_risk"
                })
            return contributions
        except Exception as e:
            logger.error(f"SHAP failed: {e}")
            return []

    def predict(self, raw_email: str) -> Dict[str, Any]:
        start_ts = time.time()
        event_id = hashlib.sha256(raw_email.encode()).hexdigest()
        
        try:
            processed = production_preprocessing(raw_email)
            if not processed:
                return {"event_id": event_id, "status": "rejected"}

            # Inference
            clean_text = processed.get("clean_text", "")
            embedding = self.ft_extractor.get_embedding(clean_text)
            final_vector = self.builder.build_vector(embedding, processed)
            prob = float(self.model.predict(xgb.DMatrix(final_vector))[0])
            
            # Thresholding
            if prob < SAFE_THRESHOLD: decision = "safe"
            elif prob < PHISHING_THRESHOLD: decision = "suspicious"
            else: decision = "phishing"

            # Expert Overrides + AI Reasons
            ai_reasons = self.explain(final_vector, processed, decision)
            heuristic_alerts = self._evaluate_security_heuristics(processed)
            
            # COMBINE AND SORT: This makes 999.x scores appear first
            all_reasons = heuristic_alerts + ai_reasons
            sorted_reasons = sorted(all_reasons, key=lambda x: abs(x["impact_score"]), reverse=True)
            if heuristic_alerts:
                decision = "phishing"
                prob = 1

            # --- VIRUS TOTAL INTEGRATION ---
            vt_results = {}
            # Only run if an API key was actually provided at startup
            if self.vt_client.headers.get("x-apikey"): 
                raw_text = processed.get("raw_text", raw_email)
                
                # 1. Extract Artifacts
                vt_urls = list(set(re.findall(r'https?://[^\s<>"\'()]+', raw_text, re.I)))
                vt_domains = set()
                vt_ips = set()
                
                # Extract Sender Domain
                from_match = re.search(r'^From:\s*.*<(.+?)>', raw_text, re.M | re.I)
                if from_match:
                    sender_dom = from_match.group(1).split('@')[-1].strip("<>\"' ").lower()
                    vt_domains.add(sender_dom)
                    
                # Extract Domains and IPs from URLs
                for link in vt_urls:
                    try:
                        netloc = urllib.parse.urlparse(link).netloc.lower().split(':')[0]
                        if netloc:
                            # Check if netloc is an IP address
                            if re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', netloc):
                                vt_ips.add(netloc)
                            else:
                                vt_domains.add(netloc)
                    except:
                        continue
                
                # 2. Fetch Reputations (multithreading)
                from concurrent.futures import ThreadPoolExecutor
                if any([vt_urls, vt_domains, vt_ips]):
                    with ThreadPoolExecutor(max_workers=5) as executor:
                    	#Create a list of tasks
        		future_urls = executor.submit(self.vt_client.get_reputations, urls=vt_urls)
        		future_doms = executor.submit(self.vt_client.get_reputations, domains=list(vt_domains))
        		future_ips = executor.submit(self.vt_client.get_reputations, ips=list(vt_ips))
        
        		# Merge the results as they finish
        		vt_results.update(future_urls.result())
        		vt_results.update(future_doms.result())
        		vt_results.update(future_ips.result())
                    

            # --- BUILD FINAL RESULT ---
            result = {
                "event_id": event_id,
                "probability": round(prob, 4),
                "decision": decision,
                "top_reasons": sorted_reasons[:5],
            }
            
            # Conditionally add reputation results ONLY if they exist
            if vt_results:
                result["reputation_results"] = vt_results
                
            result["metadata"] = {"runtime_sec": round(time.time() - start_ts, 3)}

            # Save to Wazuh JSONL
            with open(WAZUH_JSONL_PATH, "a") as f:
                f.write(json.dumps(result) + "\n")

            return result

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {"event_id": event_id, "status": "error", "message": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vt-key", help="VirusTotal API Key")
    args, _ = parser.parse_known_args()
    
    # 1. Boot up the predictor ONCE (This takes 5 minutes)
    print("\n\t [Booting up PhishGuard... Please wait]\n \n ")
    predictor = PhishGuardPredictor(vt_api_key=args.vt_key)
    print("\n\t [System Ready! Models loaded into RAM]")

    # 2. Stay open and accept emails continuously
    while True:
        print("\n\t--- Enter Email Raw Text (Type 'exit' to quit) ---")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "exit":
                    sys.exit(0)
                    
                lines.append(line)
            except EOFError:
                break # Catches Ctrl+D
                
        raw_input = "\n".join(lines)
        if not raw_input.strip(): 
            continue

        # This will now take < 1 second!
        output = predictor.predict(raw_input)
        print("\n \t--- RESULTS ---")
        print(json.dumps(output, indent=2, ensure_ascii=False))
