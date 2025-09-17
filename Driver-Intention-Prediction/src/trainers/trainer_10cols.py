# Load Packages
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import os

###########################################
# Tunable hyperparameters for the 10-feature setup
###########################################
BATCH_SIZE = 45          # Larger batch for faster epochs
NUM_EPOCHS = 190         # More epochs for convergence
DROPOUT    = 0.2         # Regularization strength

###########################################
# Load flattened CSVs:
# - expanded_data.csv: [samples*steps, 10]
# - out_data.csv:      [samples, 3]
###########################################
dataset  = np.loadtxt("expanded_data.csv", delimiter=",")
dataset1 = np.loadtxt("out_data.csv", delimiter=",")

XX = dataset
YY = dataset1

############################################
# Geometry: 20 steps × 10 features = 200-wide vector per sequence
############################################
ancho    = 20 * 10
alto     = 3000
columnas = 3

############################################
# Reshape into [samples, ancho] and [samples, classes]
############################################
data   = np.array(XX, dtype=float)
target = np.array(YY, dtype=float)

data   = np.reshape(data, (alto, ancho))         # [3000, 200]
target = np.reshape(target, (alto, columnas))    # [3000, 3]

###########################################
# Scale to [0,1] and adapt to LSTM input: [batch, timesteps=1, features=ancho]
###########################################
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
data = np.reshape(data, (alto, 1, ancho))
target = np.reshape(target, (target.shape[0], target.shape[1]))

##########################################
# Sanity prints for shapes
##########################################
print(data.shape)    # (3000, 1, 200)
print(target.shape)  # (3000, 3)

###########################################
# Split 50/50 for quick experimentation
###########################################
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.5, random_state=30
)
print(x_train.shape)
print(x_test.shape)

###########################################
# Define an LSTM classifier tuned for 10 features × 20 steps
###########################################
model = Sequential()
model.add(LSTM(160, batch_input_shape=(None, 1, ancho)))      # Wider hidden size for higher dimensional input
model.add(Dropout(DROPOUT))
model.add(Dense(10, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(DROPOUT))
model.add(Dense(3,  kernel_initializer='normal', activation='softmax'))  # 3 classes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

############################################
# Summary and training loop with validation on test split
############################################
print(model.summary())
history = model.fit(
    x_train, y_train,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test)
)

######################################################
# Persist learning curves to PNG
######################################################
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].set_title("Accuracy")
ax[0].plot(history.history.get("accuracy", []), label="Train")
ax[0].plot(history.history.get("val_accuracy", []), label="Validation")
ax[0].legend(loc="best"); ax[0].grid(True, alpha=0.3)

ax[1].set_title("Loss")
ax[1].plot(history.history.get("loss", []), label="Train")
ax[1].plot(history.history.get("val_loss", []), label="Validation")
ax[1].legend(loc="best"); ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_lstm_10cols.png", dpi=150, bbox_inches='tight')
plt.close()

#############################################
# Final metrics + predictions on test
#############################################
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predicciones = model.predict(x_test)

# Compact metrics block for the paper/repo tables
y_pred_prob = predicciones
y_pred_lbl  = np.argmax(y_pred_prob, axis=1)
y_test_lbl  = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_lbl, y_pred_lbl)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test_lbl, y_pred_lbl, average='macro', zero_division=0)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    y_test_lbl, y_pred_lbl, average='weighted', zero_division=0)

# ROC-AUC (macro) — OVR/OVO; requires one-hot labels and probabilities
classes_all = np.arange(3)
y_test_bin = label_binarize(y_test_lbl, classes=classes_all)
try:
    roc_ovr = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr', average='macro')
except Exception:
    roc_ovr = np.nan
try:
    roc_ovo = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovo', average='macro')
except Exception:
    roc_ovo = np.nan

print(f"[REPORT] LSTM 10col\n{classification_report(y_test_lbl, y_pr
