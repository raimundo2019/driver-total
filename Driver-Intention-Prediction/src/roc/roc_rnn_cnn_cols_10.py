# -*- coding: utf-8 -*-
"""
ROC for RNN and CNN using blocked time windows (STEPS=20) and 10 features.

Assumptions:
 - X: expanded_data.csv  (rows: steps*samples, cols: 10)
 - y: out_data.csv       (rows: samples,      cols: 3 one-hot classes)

Pipeline:
  1) reshape raw CSVs into [samples, steps, features]
  2) train an RNN (BiLSTM) and a 1D-CNN
  3) compute per-class ROC curves and confusion matrices for both models
"""

# ---- Standard stack ----
import argparse                          # Command-line arguments for reproducibility
import numpy as np                       # Vectorized numeric ops
import pandas as pd                      # CSV I/O
import matplotlib.pyplot as plt          # ROC and confusion plots

# ---- ML metrics ----
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import itertools                         # To place numbers inside the confusion matrix

# ---- Deep learning (Keras / TF) ----
import tensorflow as tf
from tensorflow.keras import layers, models

def build_sequences_blocked(df_features: pd.DataFrame, df_labels: pd.DataFrame, steps: int):
    """
    Convert flat, row-wise time series into sequences of length `steps`.

    Inputs
    ------
    df_features : DataFrame of shape [steps*samples, n_features]  (here n_features=10)
    df_labels   : DataFrame of shape [samples, n_classes] or [steps*samples, n_classes]
                  If labels are per-timestep, the label of the last step is used.

    Returns
    -------
    X_seq : np.ndarray, shape [samples, steps, n_features]
    Y_seq : np.ndarray, shape [samples, n_classes]
    """
    X = df_features.to_numpy(dtype=np.float32)
    y = df_labels.to_numpy(dtype=np.float32)

    N = X.shape[0]
    num_blocks = N // steps
    cut = num_blocks * steps

    # [samples, steps, features]
    X_trim = X[:cut].reshape(num_blocks, steps, X.shape[1])

    if y.shape[0] == num_blocks:
        Y_seq = y[:num_blocks]
    else:
        # step-wise labels → take last step as sequence label
        y_trim = y[:cut].reshape(num_blocks, steps, y.shape[1])
        Y_seq = y_trim[:, -1, :]

    return X_trim.astype(np.float32), Y_seq.astype(np.float32)

def plot_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, out_png: str):
    """
    One-vs-rest ROC per class.
    y_true:  one-hot array [n_samples, n_classes]
    y_score: predicted probas [n_samples, n_classes]
    """
    n_classes = y_true.shape[1]
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        auc_i = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={auc_i:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_confusion(cm: np.ndarray, class_names, title: str, out_png: str):
    """
    Render a confusion matrix with per-cell integer counts.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:d}",
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def make_rnn(input_shape, n_classes: int) -> tf.keras.Model:
    """
    BiLSTM classifier returning one label per sequence.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),                         # (steps, 10)
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def make_cnn(input_shape, n_classes: int) -> tf.keras.Model:
    """
    1D CNN over temporal dimension + GAP + dense head.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),                         # (steps, 10)
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # ---- CLI for reproducibility ----
    p = argparse.ArgumentParser()
    p.add_argument('--csv_x', type=str, default='expanded_data.csv', help='Features CSV (10 columns expected).')
    p.add_argument('--csv_y', type=str, default='out_data.csv',       help='Labels CSV (3 one-hot columns).')
    p.add_argument('--steps', type=int, default=20,                    help='Timesteps per sequence.')
    p.add_argument('--epochs', type=int, default=15,                   help='Training epochs.')
    p.add_argument('--batch_size', type=int, default=64,               help='Mini-batch size.')
    args = p.parse_args()

    # ---- Load CSVs ----
    df_X = pd.read_csv(args.csv_x)     # expects 10 feature columns
    df_y = pd.read_csv(args.csv_y)     # expects 3 one-hot columns

    # ---- Build sequences [samples, steps, features] ----
    X_all, Y_all = build_sequences_blocked(df_X, df_y, steps=args.steps)
    print("X_all:", X_all.shape, "Y_all:", Y_all.shape)  # e.g., (3000, 20, 10) and (3000, 3)

    # ---- Train/test split (stratified by one-hot labels) ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=42, stratify=Y_all
    )
    input_shape = X_tr.shape[1:]    # (steps, features) = (20, 10)
    n_classes   = y_tr.shape[1]     # 3

    # ===== RNN branch =====
    rnn = make_rnn(input_shape, n_classes)
    rnn.fit(X_tr, y_tr, validation_split=0.2, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
    y_score_rnn = rnn.predict(X_te, verbose=0)       # probabilities
    y_pred_rnn  = np.argmax(y_score_rnn, axis=1)     # predicted classes
    y_true      = np.argmax(y_te, axis=1)            # true classes

    cm_rnn = confusion_matrix(y_true, y_pred_rnn)
    print("RNN Classification report:\n", classification_report(y_true, y_pred_rnn))
    plot_confusion(cm_rnn, [f"C{i}" for i in range(n_classes)], "Confusion Matrix - RNN (10 cols)", "confusion_rnn_10.png")
    plot_roc(y_te, y_score_rnn, "Per-class ROC - RNN (10 cols)", "roc_rnn_10.png")

    # ===== CNN branch =====
    cnn = make_cnn(input_shape, n_classes)
    cnn.fit(X_tr, y_tr, validation_split=0.2, epochs=args.epochs, batch_size=args.batch_size, verbose=1)
    y_score_cnn = cnn.predict(X_te, verbose=0)
    y_pred_cnn  = np.argmax(y_score_cnn, axis=1)

    cm_cnn = confusion_matrix(y_true, y_pred_cnn)
    print("CNN Classification report:\n", classification_report(y_true, y_pred_cnn))
    plot_confusion(cm_cnn, [f"C{i}" for i in range(n_classes)], "Confusion Matrix - CNN (10 cols)", "confusion_cnn_10.png")
    plot_roc(y_te, y_score_cnn, "Per-class ROC - CNN (10 cols)", "roc_cnn_10.png")

if __name__ == "__main__":
    main()
