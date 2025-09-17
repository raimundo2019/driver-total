# -*- coding: utf-8 -*-
"""
Classical baselines (10 columns): Logistic Regression and Decision Tree.

Data geometry:
 - expanded_data.csv : [samples*steps, 10]   (flattened sequences)
 - out_data.csv      : [samples, 3]          (one-hot labels: 3 classes)

This script:
  1) reshapes features to [samples, steps*features] (single flat vector per sample),
  2) trains Logistic Regression (one-vs-rest) and Decision Tree,
  3) reports Accuracy, Precision/Recall/F1, Confusion Matrix,
  4) (optional) computes ROC-AUC macro using predicted probabilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# -----------------------------
# Load CSVs (no headers)
# -----------------------------
X_raw = np.loadtxt("expanded_data.csv", delimiter=",")  # [steps*samples, 10]
Y_raw = np.loadtxt("out_data.csv", delimiter=",")       # [samples, 3] one-hot

# -----------------------------
# Reshape to [samples, window]
# steps inferred as len(X_raw) // len(Y_raw)
# -----------------------------
n_samples = Y_raw.shape[0]
steps = X_raw.shape[0] // n_samples
n_features = X_raw.shape[1]            # expected: 10
flat_dim = steps * n_features          # e.g., 20 * 10 = 200

X = X_raw.reshape(n_samples, flat_dim) # [samples, 200]
y_onehot = Y_raw.astype(int)           # [samples, 3]
y = np.argmax(y_onehot, axis=1)        # integer labels 0..2

# -----------------------------
# Train/test split + scaling
# -----------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# -----------------------------
# Logistic Regression (OVR)
# -----------------------------
logreg = LogisticRegression(
    multi_class='ovr',    # One-vs-rest scheme
    solver='liblinear',
    max_iter=200
)
logreg.fit(X_tr, y_tr)
y_pred_lr = logreg.predict(X_te)
y_proba_lr = logreg.predict_proba(X_te)

# -----------------------------
# Decision Tree baseline
# -----------------------------
tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=None,
    random_state=42
)
tree.fit(X_tr, y_tr)
y_pred_tree = tree.predict(X_te)
try:
    y_proba_tree = tree.predict_proba(X_te)
except Exception:
    y_proba_tree = None

# -----------------------------
# Metrics helper (prints + confusion + AUC macro)
# -----------------------------
def report_block(name, y_true, y_pred, y_proba=None):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=[f"C{i}" for i in range(3)]))

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    print(f"[METRICS] {name} | acc={acc:.4f} | "
          f"Pmacro={p_macro:.4f} Rmacro={r_macro:.4f} F1macro={f1_macro:.4f} | "
          f"Pweighted={p_w:.4f} Rweighted={r_w:.4f} F1weighted={f1_w:.4f}")

    # Confusion matrix PNG
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"C{i}" for i in range(3)])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    plt.title(f"Confusion Matrix - {name} (10 cols)")
    plt.tight_layout()
    plt.savefig(f"confusion_{name.lower().replace(' ', '_')}_10cols.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Macro ROC-AUC (OVR) if probabilities are available
    if y_proba is not None:
        y_true_bin = np.eye(3)[y_true]
        try:
            auc_ovr = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro')
            print(f"[AUC-OVR macro] {name}: {auc_ovr:.4f}")
        except Exception:
            pass

# -----------------------------
# Emit reports
# -----------------------------
report_block("Logistic Regression (OVR)", y_te, y_pred_lr, y_proba_lr)
report_block("Decision Tree",            y_te, y_pred_tree, y_proba_tree)
