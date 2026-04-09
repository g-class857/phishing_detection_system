# 🛡️ PhishGuard

PhishGuard is an advanced, machine learning-powered phishing email detection engine. It combines high-speed XGBoost classification, deep linguistic embeddings (FastText), and aggressive security heuristics to detect malicious intent, brand impersonation, and homoglyph deception in real-time. 

Designed for transparency and enterprise integration, PhishGuard utilizes SHAP (SHapley Additive exPlanations) to explain its AI decisions and formats its output for seamless ingestion into SIEM dashboards like Wazuh.

## ✨ Key Features

* **Hybrid Detection Engine:** Uses an XGBoost model trained on FastText embeddings alongside a deterministic heuristic engine to evaluate raw email text.
* **Aggressive Artifact Extraction:** Automatically parses URLs, domains, and IP addresses, even when hidden behind visual tricks or raw IP hosts.
* **Homoglyph & Brand Impersonation Detection:** Normalizes Cyrillic/Greek characters and calculates Levenshtein distances to catch typosquatting (e.g., `аррle.com` or `pay-pal-security.com`).
* **Threat Intelligence Integration:** Features a threaded, asynchronous VirusTotal API client to fetch real-time reputation scores for extracted artifacts.
* **Explainable AI (XAI):** Implements SHAP to provide a human-readable "Top Reasons" list for why an email was flagged (e.g., "Urgent Words Count", "Brand Impersonation").
* **SIEM Ready:** Outputs fully structured JSONL logs containing event IDs, probabilities, decisions, and metadata, optimized for Wazuh / OpenSearch pipelines.

## Creat and activate virtual environment
python3 -m venv venv

source venv/bin/activate  
# On Windows use: .\venv\Scripts\activate

pip install -r requireements.txt 

## Usage 
python predictor.py --vt-key YOUR_VIRUSTOTAL_API_KEY

Enter the raw email and press enter then CTRL+D
