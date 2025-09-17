# Load Packages
import numpy as np                               # Numeric arrays & vector ops
import matplotlib.pyplot as plt                  # Training curves (accuracy/loss)
from keras.models import Sequential              # Keras sequential model API
from keras.layers import Dense, Dropout, LSTM    # Core LSTM classifier + MLP head
from keras.models import model_from_json         # (Optional) JSON/H5 model I/O
from sklearn.preprocessing import MinMaxScaler   # Feature scaling to [0,1]
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.preprocessing import label_binarize # For multi-class ROC
import pandas as pd                              # (Optional) data handling
import os                                        # (Optional) filesystem ops

###########################################
# Training hyperparameters (tunable)
###########################################
BATCH_SIZE = 16      # Mini-batch size
NUM_EPOCHS = 20      # Number of epochs
DROPOUT = 0.09       # Dropout regularization (0..1)

###########################################
# Load flattened CSV matrices (9 features × 20 steps per sequence)
###########################################
dataset  = np.loadtxt("expanded_data.csv", delimiter=",")  # [samples*steps, 9]
dataset1 = np.loadtxt("out_data.csv", delimiter=",")       # [samples, 3] one-hot

###########################################
# Keep original variable names as references
###########################################
XX = dataset
YY = dataset1

###########################################
# Geometry for the 9-feature setup:
# - ancho = steps*features = 20*9
# - alto  = number of sequences (3000)
# - columnas = number of classes (3)
###########################################
ancho = 20 * 9
alto = 3000
columnas = 3

###########################################
# Prepare data arrays and reshape to [samples, ancho]
###########################################
data = np.array(XX, dtype=float)
target = np.array(YY, dtype=float)

data = np.reshape(data, (alto, ancho))        # [3000, 180]
target = np.reshape(target, (alto, columnas)) # [3000, 3]

###########################################
# Scale features to [0,1] and reshape for LSTM: [batch, timesteps, features]
# Here we use a single timestep with a long feature vector (ancho).
###########################################
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
data = np.reshape(data, (alto, 1, ancho))
target = np.reshape(target, (target.shape[0], target.shape[1]))

###########################################
# Print sanity shapes for quick verification
###########################################
print(data.shape)   # Expected: (3000, 1, 180)
print(target.shape) # Expected: (3000, 3)

###########################################
# Train/test split (50/50) for quick iteration
###########################################
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.50, random_state=50
)
print(x_train.shape)  # e.g., (1500, 1, 180)
print(x_test.shape)   # e.g., (1500, 1, 180)

###########################################
# Define a compact LSTM classifier for 3-way softmax
###########################################
model = Sequential()
model.add(LSTM(80, batch_input_shape=(None, 1, ancho)))  # LSTM over a single timestep
model.add(Dropout(DROPOUT))                               # Regularization
model.add(Dense(25, kernel_initializer='normal', activation='sigmoid')) # Hidden layer
model.add(Dropout(DROPOUT))
model.add(Dense(3, kernel_initializer='normal', activation='softmax'))  # 3-class output
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())  # Architecture overview

###########################################
# Train with validation on the held-out test split
###########################################
history = model.fit(
    x_train, y_train,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test)
)

###########################################
# Save accuracy and loss curves to PNG
###########################################
fig, ax = plt.subplots(1, 2, figsize=(12,4))

ax[0].set_title("Accuracy")
ax[0].plot(history.history.get("accuracy", []), label="Train")
ax[0].plot(history.history.get("val_accuracy", []), label="Validation")
ax[0].legend(loc="best"); ax[0].grid(True, alpha=0.3)

ax[1].set_title("Loss")
ax[1].plot(history.history.get("loss", []), label="Train")
ax[1].plot(history.history.get("val_loss", []), label="Validation")
ax[1].legend(loc="best"); ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_lstm.png", dpi=150, bbox_inches='tight')
plt.close()

###########################################
# Final evaluation + predictions for metrics
###########################################
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predicciones = model.predict(x_test)

# Probabilities and hard labels
y_pred_prob = predicciones
y_pred_lbl  = np.argmax(y_pred_prob, axis=1)
y_test_lbl  = np.argmax(y_test, axis=1)

###########################################
# Full reviewer-friendly metrics (macro & weighted)
###########################################
acc = accuracy_score(y_test_lbl, y_pred_lbl)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test_lbl, y_pred_lbl, average='macro', zero_division=0)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    y_test_lbl, y_pred_lbl, average='weighted', zero_division=0)

# Multi-class ROC-AUC (macro), both OVR and OVO strategies
classes_all = np.arange(target.shape[1])
y_test_bin = label_binarize(y_test_lbl, classes=classes_all)
try:
    roc_ovr = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr', average='macro')
except Exception:
    roc_ovr = np.nan
try:
    roc_ovo = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovo', average='macro')
except Exception:
    roc_ovo = np.nan

print(f"[REPORT] LSTM 9col\n{classification_report(y_test_lbl, y_pred_lbl)}")
print(f"[METRICS] lstm_simple | acc={acc:.4f} | "
      f"Pmacro={prec_macro:.4f} Rmacro={rec_macro:.4f} F1macro={f1_macro:.4f} | "
      f"Pweighted={prec_w:.4f} Rweighted={rec_w:.4f} F1weighted={f1_w:.4f} | "
      f"AUC-OVR={roc_ovr:.4f} AUC-OVO={roc_ovo:.4f}")

###########################################
# Confusion matrix PNG for the paper/repo
###########################################
cm = confusion_matrix(y_test_lbl, y_pred_lbl)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig_cm, ax_cm = plt.subplots(figsize=(5.5,5))
disp.plot(ax=ax_cm, cmap='Blues', values_format='d', colorbar=False)
plt.title("Confusion Matrix - LSTM (9 columns)")
plt.tight_layout()
plt.savefig("confusion_lstm_9cols.png", dpi=150, bbox_inches='tight')
plt.close()
