
# -*- coding: utf-8 -*-
"""
Entrenador CNN, RNN (BiLSTM) y Híbrido (Conv1D + BiLSTM) con visualizaciones.
Carga dos CSV:
  - X: expanded_data.csv
  - Y: out_data.csv  (one-hot al final: n_clases columnas)

Salidas que genera:
  - training_curves_cnn.png
  - training_curves_rnn.png
  - training_curves_hibrido.png
  - confusion_cnn.png
  - confusion_rnn.png
  - confusion_hibrido.png
  - modelos: cnn_model.keras, rnn_model.keras, hibrido_model.keras

Uso:
  python3 trainer_cnn_rnn_auto_viz.py
"""

import os
import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# -------------------------------
# Configuración
# -------------------------------
X_CSV = "expanded_data.csv"
Y_CSV = "out_data.csv"
TEST_SIZE = 0.2
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

# -------------------------------
# Utilidades de visualización
# -------------------------------
def plot_training_curves(history, title, outfile):
    """Guarda pérdida y accuracy (train/val) en una misma figura."""
    hist = history.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # Loss
    ax[0].plot(hist.get('loss', []), label='loss')
    if 'val_loss' in hist: ax[0].plot(hist['val_loss'], label='val_loss')
    ax[0].set_title(f'Loss - {title}')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()
    # Accuracy
    # puede ser 'accuracy' o 'categorical_accuracy' según la versión
    acc_key = 'accuracy' if 'accuracy' in hist else ('categorical_accuracy' if 'categorical_accuracy' in hist else None)
    if acc_key is not None:
        ax[1].plot(hist.get(acc_key, []), label=acc_key)
        val_acc_key = f'val_{acc_key}'
        if val_acc_key in hist: ax[1].plot(hist[val_acc_key], label=val_acc_key)
        ax[1].set_title(f'Accuracy - {title}')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()
    else:
        ax[1].axis('off')
        ax[1].text(0.5, 0.5, "Accuracy no disponible", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion(y_true, y_pred, class_names, title, outfile):
    """Confusion matrix nice."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

# -------------------------------
# Carga y armado de datos
# -------------------------------
if not os.path.exists(X_CSV) or not os.path.exists(Y_CSV):
    raise FileNotFoundError(f"No se encontraron '{X_CSV}' y/o '{Y_CSV}' en el directorio actual.")

X_df = pd.read_csv(X_CSV, header=None)
Y_df = pd.read_csv(Y_CSV, header=None)

if len(X_df) % len(Y_df) != 0:
    raise ValueError(f"Las filas de X ({len(X_df)}) no son múltiplo de las filas de Y ({len(Y_df)}). "
                     "No puedo hacer reshaping [samples, STEPS, features].")

STEPS = len(X_df) // len(Y_df)
n_samples = len(Y_df)
n_features = X_df.shape[1]
n_classes = Y_df.shape[1]

print(f"[INFO] Detectado STEPS={STEPS}  | muestras={n_samples}  | features={n_features}  | clases={n_classes}")

# Escalado por columna (flatten) antes de rearmar secuencias
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df.values)

# Reorganiza a [samples, steps, features]
X_seq = X_scaled.reshape(n_samples, STEPS, n_features).astype(np.float32)
Y = Y_df.values.astype(np.float32)

# División train/test. Usamos la clase (argmax) para estratificar
y_labels = np.argmax(Y, axis=1)
X_train, X_test, Y_train, Y_test, y_train_lbl, y_test_lbl = train_test_split(
    X_seq, Y, y_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_labels
)

input_shape = X_train.shape[1:]  # (steps, features)

# -------------------------------
# Definición de modelos
# -------------------------------
def build_cnn(input_shape, n_classes):
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

def build_rnn(input_shape, n_classes):
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

def build_hibrido(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out, name="cnn_bilstm_hibrido")
    opt = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# Entrenamiento
# -------------------------------
def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, y_test_lbl, tag):
    print(f"\n[TRAIN] {tag}")
    history = model.fit(
        X_train, Y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Curvas
    plot_training_curves(history, title=tag, outfile=f"training_curves_{tag}.png")

    # Predicciones y métricas
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Confusión
    class_names = [f"C{i}" for i in range(n_classes)]
    plot_confusion(y_test_lbl, y_pred, class_names, title=f"Matriz de confusión - {tag}", outfile=f"confusion_{tag}.png")

    # Reporte en consola
    print(f"\n[REPORT] {tag}")
    print(classification_report(y_test_lbl, y_pred, target_names=class_names))

    # ===== Métricas globales requeridas por revisores =====
    acc = accuracy_score(y_test_lbl, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test_lbl, y_pred, average='macro', zero_division=0)
    prec_w, rec_w, f1_w, _            = precision_recall_fscore_support(y_test_lbl, y_pred, average='weighted', zero_division=0)

    try:
        classes_all = np.arange(n_classes)
        y_test_bin = label_binarize(y_test_lbl, classes=classes_all)
        roc_ovr = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr', average='macro')
    except Exception:
        roc_ovr = np.nan
    try:
        classes_all = np.arange(n_classes)
        y_test_bin = label_binarize(y_test_lbl, classes=classes_all)
        roc_ovo = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovo', average='macro')
    except Exception:
        roc_ovo = np.nan

    print(f"[METRICS] {tag} | acc={acc:.4f} | Pmacro={prec_macro:.4f} Rmacro={rec_macro:.4f} F1macro={f1_macro:.4f} | Pweighted={prec_w:.4f} Rweighted={rec_w:.4f} F1weighted={f1_w:.4f} | AUC-OVR={roc_ovr:.4f} AUC-OVO={roc_ovo:.4f}")

    # ===== Guardar resumen acumulado =====
    row = {
        "modelo": tag,
        "features": n_features,
        "steps": STEPS,
        "clases": n_classes,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "roc_auc_ovr_macro": roc_ovr,
        "roc_auc_ovo_macro": roc_ovo
    }
    df_out = pd.DataFrame([row])
    OUTCSV = "metricas_resumen.csv"
    if os.path.exists(OUTCSV):
        df_out.to_csv(OUTCSV, mode='a', header=False, index=False)
    else:
        df_out.to_csv(OUTCSV, index=False)


    # Guardar modelo en formato moderno
    model.save(f"{tag}_model.keras")
    print(f"[SAVE] Modelo guardado: {tag}_model.keras")

# Construye
cnn = build_cnn(input_shape, n_classes)
rnn = build_rnn(input_shape, n_classes)
hibrido = build_hibrido(input_shape, n_classes)

# Entrena y evalúa
train_and_evaluate(cnn, X_train, Y_train, X_test, Y_test, y_test_lbl, tag="cnn")
train_and_evaluate(rnn, X_train, Y_train, X_test, Y_test, y_test_lbl, tag="rnn")
train_and_evaluate(hibrido, X_train, Y_train, X_test, Y_test, y_test_lbl, tag="hibrido")

print("\n[OK] Listo. Se generaron las curvas y matrices de confusión.")
