#!/usr/bin/env python3
"""
PhishGuard XGBoost Training Pipeline
Production-grade training pipeline with:

• Optuna hyperparameter tuning
• Native XGBoost cross validation
• multiprocessing
• efficient memory usage
• early stopping
• SHAP background sampling
"""

import json # saves model metadata
import time 
import multiprocessing as mp
from pathlib import Path # object oriented path handling 

import numpy as np # array operations
import xgboost as xgb # XGBOOST algo lib for training and cross validation
import optuna # hyperparameter opeimisation framework 
from optuna.pruners import MedianPruner # prunes unpromising trials early
from sklearn.metrics import average_precision_score # calculate precision score after training
import joblib # saves scikit-learn wrapper model 
import gc  # garbage collector to free memory after each trial
from sklearn.model_selection import train_test_split # split data to train/test
from sklearn.metrics import precision_recall_curve # compute optimal threshold from precision-recall curve
# ==========================================================
# Paths
# ==========================================================
ROOT= Path(__file__).parent.parent
DATA_PATH = ROOT / "data/processed/embed_output/xgboost_features.npz"

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "phishguard_xgb.json" # native model 
METADATA_PATH = MODEL_DIR / "model_metadata.json" # training metadata
SHAP_BACKGROUND_PATH = MODEL_DIR / "shap_background.npy" # background sample for SHAP explanations 
# ==========================================================
# Global settings
# ==========================================================
RANDOM_STATE = 42 # for reproducibility
N_THREADS = 2 # CPU threads 
N_FOLDS = 3 # folds number used in cross validation

N_TRIALS = 50 # number of optuna trials 
NUM_BOOST_ROUND = 2000 # maximum boosting rounds for final training
EARLY_STOPPING = 50 # stop after 50 rounds if there's no improvements
# ==========================================================
# Load dataset (memory efficient)
# ==========================================================
print("Loading dataset...")

npz = np.load(DATA_PATH, mmap_mode="r") # open file in memory mapped read=only on demand not fully loaded
X = np.array(npz["X"], dtype=np.float32)
y = np.array(npz["y"], dtype=np.int32)
# Forces loading into memory as actual numpy arrays (removes the memory‑mapped reference). This is fine because after filtering we will have a manageable size.

mask = (y != -1) # remove rows with label -1
X = X[mask]
y = y[mask]

print("Dataset shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y, # ensures the same class distribution in both splits
    random_state=42
)
# ==========================================================
# Class imbalance handling
# ==========================================================
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / pos # negative to positive ratio

print("Positives:", pos)
print("Negatives:", neg)
print("scale_pos_weight:", scale_pos_weight) # tells XGBoost to weight positive class more heavily when >1.
# ==========================================================
# Base XGBoost parameters
# ==========================================================
BASE_PARAMS = {
    "objective": "binary:logistic", # binary classification, output probability.
    "eval_metric": "aucpr", # optimise area under precision‑recall curve (good for imbalanced data).
    "tree_method": "hist", # histogram‑based algorithm (faster, lower memory).
    
    "grow_policy": "lossguide", #  grow tree by splitting leaf with largest loss reduction (instead of depth‑wise).
    "verbosity": 0, # silent
    "nthread": N_THREADS,
    
    "max_bin": 128, # histogtam bins 
    "scale_pos_weight": scale_pos_weight # as computed above
}
# ==========================================================
# Optuna objective with native XGBoost CV
# ==========================================================
# ---------- CONFIG FOR SAFE TUNING ----------
N_TRIALS = 50                       # total optuna trials for tuning
TUNE_SUBSET_MAX = 60000             # use at most this many rows for tuning (adjust to memory)
TUNE_NUM_BOOST_ROUND = 600          # lower rounds during tuning
TUNE_EARLY_STOP = 30                # stop after 30 rounds if not improved
OPTUNA_TIMEOUT_SECONDS = 60 * 60 * 4  # 3 hours max for tuning run (optional)
# ------------------------------------------------
def objective(trial): # called by optuna for each huperparameter combination
    # =========================
    # Sample subset (IMPORTANT)
    # =========================
    rng = np.random.RandomState(42 + trial.number) # use different random seed for trial
    subset_size = min(40000, X_train.shape[0]) # takes at mose 40,000 rows 
    idx = rng.choice(X_train.shape[0], subset_size, replace=False) # randomly samples without replacement
    # this reduces tuning time and memory
    X_sub = X_train[idx]
    y_sub = y_train[idx] 
    # =========================
    # Define parameters
    # =========================
    params = BASE_PARAMS.copy()

    params.update({
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True), # samples on logarithmic scale, float for (learning rate, regularization)
        "max_depth": trial.suggest_int("max_depth", 4, 10), 
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), # fraction of rows used per tree 
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), # fraction of features used per tree
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15), # minimum sum of instance weights in a child 
        "gamma": trial.suggest_float("gamma", 0.0, 5.0), # minimum loss reduction required to split
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True), # L2 regularization
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True), # L1 regularization
    })
    # =========================
    # Create DMatrix
    # =========================
    dtrain = xgb.DMatrix(X_sub, label=y_sub, nthread=2) # xgboost internal data structure
    # =========================
    # Cross Validation
    # =========================
    cv = xgb.cv( # performs cross‑validation and returns a DataFrame of metrics.
        params=params,
        dtrain=dtrain,
        num_boost_round=600, # uses the tuning‑specific lower value.
        nfold=3,
        stratified=True, # preserves class proportions in each fold.
        early_stopping_rounds=25,
        seed=42,
        verbose_eval=False
    )

    best_score = cv["test-aucpr-mean"].max() 
    #  column with best value of mean AUC‑PR across folds 
    # =========================
    # Memory cleanup (CRITICAL)
    # =========================
    del dtrain
    del cv
    gc.collect()
    # manual deletion and garbage collection to prevent memory buildup
    return float(best_score)
# create study with pruning
study = optuna.create_study(
    study_name="phishguard_tune",
    storage="sqlite:///optuna_phishguard.db", # save study result for persistence in sqlite db
    load_if_exists= True, # resumes from previous  
    direction="maximize", # we wanna maximize 
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE), # Tree‑structured Parzen Estimator (default, good for hyperparameter search).
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5), # prunes trials that perform below the median of completed trials after 5 warm‑up steps.
)
best_params = study.best_trial.params
print("Best parameters (from subset tuning):", best_params)

# ==========================================================
# Run hyperparameter tuning
# ==========================================================

print("Starting Optuna tuning...")
# run single-process optuna to avoid massive parallel memory usage
study.optimize(
    objective,
    n_jobs=1, # runs trials sequentially (avoids memory explosion).
    n_trials=N_TRIALS,
    timeout=60*60*4, 
    gc_after_trial=True
)
best_params = study.best_trial.params 
print("Best parameters:", best_params)

# ==========================================================
# Train final model
# ==========================================================
final_params = BASE_PARAMS.copy()
final_params.update(best_params)# Merge best parameters into base parameters.
final_params["scale_pos_weight"]= scale_pos_weight # in case if overwritten
dtrain = xgb.DMatrix(X_train, label=y_train)
print("Training final model...")

model = xgb.train(
    final_params, 
    dtrain,# use all training data 
    num_boost_round=2000,# maximum boosting rounds 
    verbose_eval=100, # print evaluation metric every 100 rounds
)
model.save_model(MODEL_PATH)
print("Model saved:", MODEL_PATH)
# ==========================================================
# Evaluation metrics
# ==========================================================

print("Evaluating model...")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

dtest = xgb.DMatrix(X_test)
y_pred_prob = model.predict(dtest) # predicts probabilities on the test set
y_labels = np.where(
    y_pred_prob < 0.3, "safe",
    np.where(y_pred_prob < 0.7, "suspicious", "phishing")
)
y_pred_binary = (y_pred_prob > 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Precision:", precision_score(y_test, y_pred_binary))
print("Recall:", recall_score(y_test, y_pred_binary))
print("F1:", f1_score(y_test, y_pred_binary))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
# F1 score for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] # finds threshold that maximize threshold, can  be used for decision making 

print("Best threshold:", best_threshold)
print("Best F1:", f1_scores[best_idx])
print("Precision at best:", precision[best_idx])
print("Recall at best:", recall[best_idx])
# ==========================================================
# Save SHAP background sample
# ==========================================================

sample_size = min(1024, X.shape[0]) # takes up to 1024 random rows from the full dataset
rng = np.random.RandomState(RANDOM_STATE)
idx = rng.choice(X.shape[0], sample_size, replace=False)
bg = X[idx]
np.save(SHAP_BACKGROUND_PATH, bg) # save as npy
print("SHAP background saved")
# ==========================================================
# Save metadata
# ==========================================================

metadata = { # record metadata, usefull for auditing and reproducibility
    "samples": int(len(y)),
    "features": int(X.shape[1]),
    "best_params": best_params,
    "scale_pos_weight": float(scale_pos_weight),
    "timestamp": time.time()
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)
# ==========================================================
# Optional sklearn wrapper (for deployment)
# ==========================================================

from xgboost import XGBClassifier
clf = XGBClassifier( # Creates a scikit‑learn compatible XGBClassifier.
    **best_params,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=N_THREADS
)
clf.fit(X_train, y_train) # trains on the same data 
joblib.dump(clf, MODEL_DIR / "phishguard_xgb.pkl")
print("Training completed successfully.")
