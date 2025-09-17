# -*- coding: utf-8 -*-
"""
Trainer for three sequence classifiers with 9 features and fixed STEPS:
 - CNN (1D temporal conv)
 - RNN (BiLSTM)
 - Hybrid (Conv1D + BiLSTM)

It loads two CSVs:
  - X: expanded_data.csv      (rows: steps*samples, cols: 9)
  - Y: out_data.csv           (rows: samples,      cols: 3 one-hot)
and produces:
  - training_curves_{tag}.png (loss/accuracy per epoch)
  - confusion_{tag}.png       (confusion matrix on test set)
  - *.keras                   (saved Keras models)
"""

import os
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D", category=UserWarning)

# ---- Core scientific stack ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Scikit-learn utilities ----
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ---- TensorFlow / Keras ----
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# -------------------------------
# Configuration hyperparameters
# -------------------------------
X_CSV = "expanded_data.csv"   # features (9 columns expected)
Y_CSV = "out_data.csv"        # labels   (3 one-hot columns expected)
TEST_SIZE = 0.2               # holdout fraction
EPOCHS = 30                   # training epochs
BATCH_SIZE = 64               # minibatch size
LEARNING_RATE = 1e-3          # Adam learning rate
RANDOM_STATE = 42             # reproducibility for split

# -------------------------------
# Plot helpers (loss + accuracy)
# -------------------------------
def plot_training_curves(history: tf.keras.callbacks.History, title: str, outfile: str):
    """Save train/val loss and accuracy curves from a Keras History object."""
    hist = history.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Loss subplot
    ax[0].plot(hist.get('loss', []), label='loss')
    if 'val_loss' in hist:
        ax[0].plot(hist['val_loss'], label='val_loss')
    ax[0].set_title(f'Loss - {title}')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Accuracy subplot (key name can vary with TF/Keras version)
    acc_key = 'accuracy' if 'accuracy' in hist else ('categorical_accuracy' if 'categorical_accuracy' in hist else None)
    if acc_key is not None:
        ax[1].plot(hist.get(acc_key, []), label=acc_key)
        val_acc_key = f'val_{acc_key}'
        if val_acc_key in hist:
            ax[1].plot(hist[val_acc_key], label=val_acc_key)
        ax[1].set_title(f'Accuracy - {title}')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()
    else:
        # If accuracy is not reported by the backend/metrics, show a placeholder
        ax[1].axis('off')
        ax[1].text(0.5, 0.5, "Accuracy not available", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names, title: str, outfile: str):
    """Render and save a confusion matrix with pretty formatting."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

# -------------------------------
# Data loading & shaping
# -------------------------------
if not os.path.exists(X_CSV) or not os.path.exists(Y_CSV):
    raise FileNotFoundError(f"Missing '{X_CSV}' and/or '{Y_CSV}' in the working directory.")

# By default the CSVs have no header; read raw numeric matrices
X_df = pd.read_csv(X_CSV, header=None)  # (steps*samples, 9)
Y_df = pd.read_csv(Y_CSV, header=None)  # (samples, 3)

# Sanity check: X rows must be a multiple of Y rows to form blocks
if len(X_df) % len(Y_df) != 0:
    raise ValueError(
        f"X rows ({len(X_df)}) is not a multiple of Y rows ({len(Y_df)}). "
        "Cannot reshape into [samples, STEPS, features]."
    )

STEPS      = len(X_df) // len(Y_df)      # infer steps (e.g., 20)
n_samples  = len(Y_df)                   # number of sequences
n_features = X_df.shape[1]               # should be 9
n_classes  = Y_df.shape[1]               # should be 3

print(f"[INFO] Detected STEPS={STEPS} | samples={n_samples} | features={n_features} | classes={n_classes}")

# Column-wise standardization on the flattened matrix before reassembling sequences
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_df.values)

# Repack to [samples, steps, features] with float32 for TensorFlow efficiency
X_seq = X_scaled.reshape(n_samples, STEPS, n_features).astype(np.float32)
Y     = Y_df.values.astype(np.float32)   # one-hot labels

# Stratified train/test using argmax labels for balancing
y_labels = np.argmax(Y, axis=1)
X_train, X_test, Y_train, Y_test, y_train_lbl, y_test_lbl = train_test_split(
    X_seq, Y, y_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_labels
)

input_shape = X_train.shape[1:]  # (steps, features) e.g., (20, 9)

# -------------------------------
# Model definitions
# -------------------------------
def build_cnn(input_shape, n_classes: int) -> tf.keras.Model:
    """1D CNN over the time dimension with GAP and dense head."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out, name="cnn_1d")
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_rnn(input_shape, n_classes: int) -> tf.keras.Model:
    """BiLSTM stack that returns a single label per sequence."""
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out, name="bilstm_rnn")
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_hibrido(input_shape, n_classes: int) -> tf.keras.Model:
    """Conv1D front-end to summarize local patterns + BiLSTM to model long-range dynamics."""
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out, name="cnn_bilstm_hybrid")
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# Train/evaluate helper
# -------------------------------
def train_and_evaluate(model: tf.keras.Model,
                       X_train, Y_train, X_test, Y_test,
                       y_test_lbl: np.ndarray, tag: str):
    """
    Generic trainer:
      - fits the model with a validation split,
      - saves training curves,
      - computes predictions on test,
      - saves confusion matrix,
      - prints a classification report and consolidated metrics
        (accuracy, macro/weighted P/R/F1, ROC-AUC OVR/OVO when possible).
    """
    print(f"\n[TRAIN] {tag}")
    history = model.fit(
        X_train, Y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # curves.png
    plot_training_curves(history, title=tag, outfile=f"training_curves_{tag}.png")

    # probs & labels
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    # confusion.png
    class_names = [f"C{i}" for i in range(Y_train.shape[1])]
    plot_confusion(y_test_lbl, y_pred, class_names, title=f"Confusion Matrix - {tag}", outfile=f"confusion_{tag}.png")

    # console report
    print(f"\n[REPORT] {tag}")
    print(classification_report(y_test_lbl, y_pred, target_names=class_names))

    # consolidated metrics (for tables in the paper/rebuttal)
    acc = accuracy_score(y_test_lbl, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test_lbl, y_pred, average='macro',   zero_division=0)
    prec_w,     rec_w,     f1_w,     _ = precision_recall_fscore_support(y_test_lbl, y_pred, average='weighted', zero_division=0)

    # ROC-AUC (macro) in OVR/OVO mode — requires one-hot / probs
    try:
        classes_all = np.arange(Y_train.shape[1])
        y_test_bin  = label_binarize(y_test_lbl, classes=classes_all)
        roc_ovr = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr', average='macro')
    except Exception:
        roc_ovr = np.nan
    try:
        classes_all = np.arange(Y_train.shape[1])
        y_test_bin  = label_binarize(y_test_lbl, classes=classes_all)
        roc_ovo = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovo', average='macro')
    except Exception:
        roc_ovo = np.nan

    print(f"[METRICS] {tag} | acc={acc:.4f} | "
          f"Pmacro={prec_macro:.4f} Rmacro={rec_macro:.4f} F1macro={f1_macro:.4f} | "
          f"Pweighted={prec_w:.4f} Rweighted={rec_w:.4f} F1weighted={f1_w:.4f} | "
          f"AUC-OVR={roc_ovr if not np.isnan(roc_ovr) else float('nan'):.4f} "
          f"AUC-OVO={roc_ovo if not np.isnan(roc_ovo) else float('nan'):.4f}")

    # save Keras model for later inference / reproducibility
    model.save(f"{tag}_model.keras")

# -------------------------------
# Orchestrate training
# -------------------------------
if __name__ == "__main__":
    # build models
    cnn     = build_cnn(input_shape, n_classes)
    rnn     = build_rnn(input_shape, n_classes)
    hybrid  = build_hibrido(input_shape, n_classes)

    # train/eval all
    train_and_evaluate(cnn,    X_train, Y_train, X_test, Y_test, y_test_lbl, tag="cnn")
    train_and_evaluate(rnn,    X_train, Y_train, X_test, Y_test, y_test_lbl, tag="rnn")
    train_and_evaluate(hybrid, X_train, Y_train, X_test, Y_test, y_test_lbl, tag="hibrido")
