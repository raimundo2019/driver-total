# Load Packages
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

###########################################
# Load datasets (10 features × 20 steps per sequence)
###########################################
dataset  = np.loadtxt("expanded_data.csv", delimiter=",")  # Features (flattened)
dataset1 = np.loadtxt("out_data.csv", delimiter=",")       # One-hot labels

XX1 = dataset
YY1 = dataset1

############################################
# Geometry for 10 features:
# - ventana = 20*10
# - filas (samples) = 3000
# - columnas (classes) = 3
############################################
ventana = 20 * 10
filas = 3000
columnas = 3

############################################
# Reshape flat matrices into [samples, ventana] and [samples, classes]
############################################
X = np.array(XX1, dtype='float32')
X = np.reshape(XX1, (filas, ventana))
Y = np.array(YY1, dtype='float32')
Y = Y.reshape((filas, columnas))

###########################################
# Scale features and split 50/50 for testing purposes
###########################################
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=30
)

# LSTM input shape: [batch, timesteps=1, features=ventana]
x_train = x_train.reshape((x_train.shape[0], 1, ventana))
x_test  = x_test.reshape((x_test.shape[0],  1, ventana))

#####################################
# Load pre-trained model (JSON + H5)
#####################################
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# Compile for evaluation
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# NOTE: consider 'categorical_crossentropy' for softmax outputs with 3 classes.

###########################################
# Predict and binarize outputs at 0.5 threshold (one-vs-rest style)
###########################################
n_classes = 3
predict = loaded_model.predict(x_test)

XX = np.zeros_like(predict, dtype='int32')
for i in range(1, predict.shape[0]):
    for j in range(1, predict.shape[1]):
        XX[i-1:i, j-1:j] = 1 if predict[i-1:i, j-1:j] >= 0.5 else 0

###########################################
# Quick accuracy and classification report
###########################################
scores = loaded_model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(XX, axis=1)
print("Classification report:")
print(classification_report(y_true, y_pred, target_names=["TR", "GS", "TL"]))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["TR", "GS", "TL"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

###########################################
# ROC per class (using hard labels XX; probs would be smoother)
###########################################
fpr = dict(); tpr = dict(); roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], XX[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), XX.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# (Macro-average code can be added similarly; omitted for brevity)
