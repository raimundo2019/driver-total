# -*- coding: utf-8 -*-
"""
ROC for RNN and CNN using blocked time windows (STEPS=20) and 9 features.
Assumptions:
 - X: expanded_data.csv (rows: steps*samples, cols: 9)
 - y: out_data.csv      (rows: samples,      cols: 3 one-hot classes)

This script:
  1) reshapes raw CSVs into [samples, steps, features],
  2) trains an RNN and a 1D-CNN,
  3) plots per-class ROC curves and confusion matrices for both models.
"""

# ---- Standard/ML/DL utilities ----
import argparse                  # Parse command-line arguments for reproducible runs
import numpy as np               # Numerical arrays and vectorized ops
import pandas as pd              # CSV I/O + tabular handling
import matplotlib.pyplot as plt  # Plotting (ROC, confusion matrix)
from sklearn.model_selection import train_test_split  # Stratified split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report  # Metrics & reports
import itertools                 # Grid iterator for drawing numbers inside confusion matrix
import tensorflow as tf          # Deep learning backend
from tensorflow.keras import layers, models  # Keras API for model definition

def build_sequences_blocked(df_features: pd.DataFrame, df_labels: pd.DataFrame, steps: int):
    """
    Convert flat rows into sequences (blocks) of length `steps`.
    - df_features: shape [steps*samples, n_features]
    - df_labels:   either [samples, n_classes]  or [steps*samples, n_classes]
      If labels are per-step, the last step in each block is used as the block label (sequence → single label).
    Returns:
      X_seq: [samples, steps, n_features]
      Y_seq: [samples, n_classes]
    """
    X = df_features.to_numpy(dtype=np.float32)         # raw features as float32
    y = df_labels.to_numpy(dtype=np.float32)           # raw labels as float32

    N = X.shape[0]                                     # total rows in feature CSV
    num_blocks = N // steps                            # number of sequences (samples)
    cut = num_blocks * steps                           # truncate to full blocks

    # reshape features into [samples, steps, features]
    X_trim = X[:cut].reshape(num_blocks, steps, X.shape[1])

    # if labels already match samples, take them directly
    if y.shape[0] == num_blocks:
        y_seq = y[:num_blocks]
    else:
        # otherwise assume step-wise labels and take the final step as the block label
        y_trim = y[:cut].reshape(num_blocks, steps, y.shape[1])
        y_seq = y_trim[:, -1, :]

    return X_trim.astype(np.float32), y_seq.astype(np.float32)

def plot_roc(y_true: np.ndarray, y_score: np.ndarray, title: str, out_png: str):
    """
    Draw one-vs-rest ROC curves per class.
    - y_true:  one-hot true labels, shape [n_samples, n_classes]
    - y_score: predicted probabilities, shape [n_samples, n_classes]
    """
    n_classes = y_true.shape[1]
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])  # compute ROC points
        roc_auc = auc(fpr, tpr)                                # compute AUC
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)     # diagonal baseline
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)                             # persist high-res plot
    plt.close()

def plot_confusion(cm: np.ndarray, class_names, title: str, out_png: str):
    """
    Pretty confusion matrix with inline integer counts.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # write counts in each cell with contrasting color
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def make_rnn(input_shape, n_classes: int) -> tf.keras.Model:
    """
    BiLSTM classifier producing a single label per sequence.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),                             # (steps, features)
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')                # 3-way softmax
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def make_cnn(input_shape, n_classes: int) -> tf.keras.Model:
    """
    1D CNN over the temporal axis + GAP (global average pooling) + MLP head.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),                             # (steps, features)
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # ---- CLI arguments ensure runs are traceable and easy to reproduce ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_x', type=str, default='expanded_data.csv', help='Path to features CSV (9 columns).')
    parser.add_argument('--csv_y', type=str, default='out_data.csv',       help='Path to one-hot labels CSV (3 columns).')
    parser.add_argument('--steps', type=int, default=20,                    help='Timesteps per sequence (block length).')
    parser.add_argument('--epochs', type=int, default=15,                   help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=64,               help='Minibatch size.')
    args = parser.parse_args()

    # ---- Load CSVs ----
    df_X = pd.read_csv(args.csv_x)   # expects 9 feature columns
    df_y = pd.read_csv(args.csv_y)   # expects 3 one-hot columns

    # ---- Basic shape checks for early failure (faster debugging) ----
    feature_cols = list(df_X.columns)
    label_cols   = list(df_y.columns)
    if len(feature_cols) != 9:
        print("WARNING: expected 9 feature columns, found:", len(feature_cols))
    if len(label_cols) != 3:
        print("WARNING: expected 3 label columns (one-hot), found:", len(label_cols))

    # ---- Turn flat rows into sequences [samples, steps, features] ----
    X_all, Y_all = build_sequences_blocked(df_X[feature_cols], df_y[label_cols], steps=args.steps)
    print("X_all:", X_all.shape, "Y_all:", Y_all.shape)

    # ---- Stratified split on class labels (argmax over one-hot) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=42, stratify=Y_all
    )

    n_classes  = y_train.shape[1]    # 3
    input_shape = X_train.shape[1:]  # (steps, features) = (20, 9)

    # ===================== RNN branch =====================
    rnn = make_rnn(input_shape, n_classes)
    hist_rnn = rnn.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    # predicted probabilities on holdout set
    y_score_rnn = rnn.predict(X_test, verbose=0)
    y_pred_rnn  = np.argmax(y_score_rnn, axis=1)   # predicted class indices
    y_true      = np.argmax(y_test, axis=1)        # true class indices

    # Confusion matrix + text report for class-level metrics
    cm_rnn = confusion_matrix(y_true, y_pred_rnn)
    print("RNN Classification report:\n", classification_report(y_true, y_pred_rnn))

    # Save diagnostic plots
    plot_confusion(cm_rnn, [f"C{i}" for i in range(n_classes)], "Confusion Matrix - RNN", "confusion_rnn.png")
    plot_roc(y_test, y_score_rnn, "Per-class ROC - RNN", "roc_rnn.png")

    # ===================== CNN branch =====================
    cnn = make_cnn(input_shape, n_classes)
    hist_cnn = cnn.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    y_score_cnn = cnn.predict(X_test, verbose=0)
    y_pred_cnn  = np.argmax(y_score_cnn, axis=1)

    cm_cnn = confusion_matrix(y_true, y_pred_cnn)
    print("CNN Classification report:\n", classification_report(y_true, y_pred_cnn))

    plot_confusion(cm_cnn, [f"C{i}" for i in range(n_classes)], "Confusion Matrix - CNN", "confusion_cnn.png")
    plot_roc(y_test, y_score_cnn, "Per-class ROC - CNN", "roc_cnn.png")

if __name__ == "__main__":
    main()
