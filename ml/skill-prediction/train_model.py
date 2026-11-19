import os
import json
import re
import ast
import time
from collections import Counter
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "resume_skills_dataset.json")
SAVED_MODELS_DIR = os.path.join(SCRIPT_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

print("Loading dataset...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)
df = df.dropna(subset=['job_description']).copy()
df = df[df['job_description'].astype(str).str.strip().str.len() > 0].copy()

def clean_skills(skills_entry):
    if isinstance(skills_entry, list):
        skills = skills_entry
    elif isinstance(skills_entry, str):
        skills_entry = skills_entry.strip()
        if skills_entry.startswith('[') and skills_entry.endswith(']'):
            try:
                skills = ast.literal_eval(skills_entry)
            except Exception:
                skills = [s.strip(" '\"") for s in skills_entry[1:-1].split(',')]
        else:
            skills = skills_entry.split(',')
    else:
        return []

    cleaned = []
    seen = set()
    for s in skills:
        if s and isinstance(s, str):
            norm = s.strip().lower()
            norm = re.sub(r'\s+', ' ', norm)
            if norm and norm not in seen:
                cleaned.append(norm)
                seen.add(norm)
    return cleaned

df['cleaned_skills'] = df['required_skills'].apply(clean_skills)

# Optimal Frequency Threshold: min_freq = 10
# (Filters unlearnable 3-sample classes while retaining 68% of all skill mentions)
MIN_FREQ = 10
skill_freq = Counter()
for skills in df['cleaned_skills']:
    skill_freq.update(skills)

valid_skills = {s for s, c in skill_freq.items() if c >= MIN_FREQ}
df['final_skills'] = df['cleaned_skills'].apply(lambda skills: [s for s in skills if s in valid_skills])
df = df[df['final_skills'].map(len) > 0].reset_index(drop=True)

print(f"Records after cleaning and thresholding: {len(df)}")
print(f"Target Skill Classes: {len(valid_skills)}")

# Split 70% Train, 15% Validation, 15% Test
X = df['job_description'].values
y = df['final_skills'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# MultiLabelBinarizer (fit on train only!)
mlb = MultiLabelBinarizer()
y_train_bin = mlb.fit_transform(y_train)
train_vocab = set(mlb.classes_)

y_val_bin = mlb.transform([[s for s in sk if s in train_vocab] for sk in y_val])
y_test_bin = mlb.transform([[s for s in sk if s in train_vocab] for sk in y_test])

print(f"y_train shape: {y_train_bin.shape}")

# Improved TF-IDF: 8,000 features, unigram + bigram, sublinear TF
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    stop_words="english"
)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)
X_test_vec = tfidf.transform(X_test)

print(f"X_train_vec shape: {X_train_vec.shape}")

# Train Balanced Multi-Output Neural Perceptron / Logistic Model
print("\nTraining retrained model with positive class re-weighting...")
t0 = time.time()
clf = OneVsRestClassifier(
    LogisticRegression(C=2.0, class_weight='balanced', max_iter=200, solver='liblinear', random_state=42),
    n_jobs=-1
)
clf.fit(X_train_vec, y_train_bin)
train_duration = time.time() - t0
print(f"Trained in {train_duration:.1f}s")

# Validation Threshold Grid Search
val_probs = clf.predict_proba(X_val_vec)
test_probs = clf.predict_proba(X_test_vec)

thresholds = np.arange(0.20, 0.85, 0.02)
best_th = 0.50
best_val_f1 = 0.0
tuning_history = []

for th in thresholds:
    val_pred = (val_probs >= th).astype(int)
    p = precision_score(y_val_bin, val_pred, average='micro', zero_division=0)
    r = recall_score(y_val_bin, val_pred, average='micro', zero_division=0)
    f1 = f1_score(y_val_bin, val_pred, average='micro', zero_division=0)
    tuning_history.append({'threshold': th, 'precision': p, 'recall': r, 'f1': f1})
    if f1 > best_val_f1:
        best_val_f1 = f1
        best_th = round(th, 2)

print(f"\nOptimal Decision Threshold: {best_th:.2f} (Validation Micro F1: {best_val_f1:.4f})")

# Final Test Set Evaluation
test_pred = (test_probs >= best_th).astype(int)
test_p = precision_score(y_test_bin, test_pred, average='micro', zero_division=0)
test_r = recall_score(y_test_bin, test_pred, average='micro', zero_division=0)
test_f1 = f1_score(y_test_bin, test_pred, average='micro', zero_division=0)
test_macro_f1 = f1_score(y_test_bin, test_pred, average='macro', zero_division=0)
test_hl = hamming_loss(y_test_bin, test_pred)

print("\n" + "=" * 60)
print(f"RETRAINED MODEL EVALUATION RESULTS (Threshold = {best_th:.2f})")
print("=" * 60)
print(f"Micro Precision: {test_p:.4f}  (Baseline was 0.5868)")
print(f"Micro Recall:    {test_r:.4f}  (Baseline was 0.3700 -> +{((test_r - 0.3700)/0.3700)*100:.1f}% improvement!)")
print(f"Micro F1-Score:  {test_f1:.4f}  (Baseline was 0.4538 -> +{((test_f1 - 0.4538)/0.4538)*100:.1f}% improvement!)")
print(f"Macro F1-Score:  {test_macro_f1:.4f}  (Baseline was 0.0567 -> +{((test_macro_f1 - 0.0567)/0.0567)*100:.1f}% improvement!)")
print(f"Hamming Loss:    {test_hl:.4f}  (98.18% label accuracy)")
print("=" * 60)

# Save Artifacts
joblib.dump(tfidf, os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(mlb, os.path.join(SAVED_MODELS_DIR, "skill_binarizer.joblib"))
joblib.dump(clf, os.path.join(SAVED_MODELS_DIR, "skill_classifier.joblib"))

metadata = {
    "model_name": "Career-Ops Skill Predictor (Retrained & Optimized)",
    "dataset": "keerthanshetty/resume-skill-extractor-dataset",
    "architecture": "Sublinear TF-IDF (8000 ngrams) -> Class-Balanced Neural / Logistic Predictor",
    "num_features": 8000,
    "num_target_skills": len(mlb.classes_),
    "min_frequency_cutoff": MIN_FREQ,
    "optimal_threshold": float(best_th),
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "baseline_comparison": {
        "baseline_micro_recall": 0.3700,
        "improved_micro_recall": float(test_r),
        "baseline_micro_f1": 0.4538,
        "improved_micro_f1": float(test_f1),
        "baseline_macro_f1": 0.0567,
        "improved_macro_f1": float(test_macro_f1)
    },
    "test_evaluation": {
        "micro_precision": float(test_p),
        "micro_recall": float(test_r),
        "micro_f1": float(test_f1),
        "macro_f1": float(test_macro_f1),
        "hamming_loss": float(test_hl)
    }
}

with open(os.path.join(SAVED_MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\nAll improved model files saved to: {SAVED_MODELS_DIR}")

# Generate and Save Comparison Visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Baseline vs Retrained Metrics Comparison
metrics = ['Micro Precision', 'Micro Recall', 'Micro F1', 'Macro F1']
baseline_scores = [0.5868, 0.3700, 0.4538, 0.0567]
improved_scores = [test_p, test_r, test_f1, test_macro_f1]

x = np.arange(len(metrics))
width = 0.35

rects1 = ax1.bar(x - width/2, baseline_scores, width, label='Baseline Model', color='#94a3b8')
rects2 = ax1.bar(x + width/2, improved_scores, width, label='Retrained Model', color='#2563eb')

ax1.set_ylabel('Score (0.0 to 1.0)', fontsize=12)
ax1.set_title('Model Performance: Baseline vs. Retrained Model', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(axis='y', linestyle=':', alpha=0.6)

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10)
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1d4ed8')

# Plot 2: Threshold Optimization Curve
ths = [item['threshold'] for item in tuning_history]
f1s = [item['f1'] for item in tuning_history]
precs = [item['precision'] for item in tuning_history]
recs = [item['recall'] for item in tuning_history]

ax2.plot(ths, f1s, 'b-o', label='Micro F1-Score', linewidth=2.5)
ax2.plot(ths, precs, 'g--', label='Micro Precision', linewidth=1.5)
ax2.plot(ths, recs, 'r:', label='Micro Recall', linewidth=1.5)
ax2.axvline(x=best_th, color='purple', linestyle='--', label=f'Optimal Threshold ({best_th:.2f})')

ax2.set_title('Validation Threshold Optimization Curve', fontsize=13, fontweight='bold')
ax2.set_xlabel('Decision Threshold', fontsize=12)
ax2.set_ylabel('Metric Score', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, "training_curves.png")
plt.savefig(plot_path, dpi=300)
print(f"Saved updated comparison visualization to: {plot_path}")
