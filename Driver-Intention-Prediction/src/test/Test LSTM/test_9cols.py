#Load Packages
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy import interp
from itertools import cycle
###########################################
###########################################
dataset  = np.loadtxt("expanded_data.csv", delimiter=",")
dataset1 = np.loadtxt("out_data.csv", delimiter=",")
###########################################
XX1 = dataset
YY1 = dataset1
############################################
ventana=20*9 ### son 10 ventanas y 7 características
filas=3000      ### son 99 ejemplos
columnas=3    ### se contempla tres salidad o posibilidades ellas son GD, GI y SD
############################################
X=np.array(XX1,'float32')
X=np.reshape(XX1,(filas,ventana))
Y=np.array(YY1,'float32')
Y=Y.reshape((filas,columnas))
###########################################
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
###########################################
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.5,random_state=50)
x_train=x_train.reshape((1500,1,ventana))
x_test =x_test.reshape((1500,1,ventana))#### X representa x_test
#####################################
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
####################################
#'binary_crossentropy'
#'categorical_crossentropy'
################################
#np.savetxt('testPredict.txt', trainPredict,fmt='%0.2f', delimiter=',')
###########################################
n_classes=3
#np.savetxt('test.out', loaded_model.predict(X),fmt='%0.3f', delimiter=',')
predict=loaded_model.predict(x_test)
#print(predict[0:36])
# evalua el modelo
XX=[]
v=0.0
XX=np.array(XX,'int32')
XX=np.reshape(predict,(predict.shape[0],predict.shape[1]))
################################################
for i in range(1,predict.shape[0]):#predict.shape[0]
    for j in range(1,predict.shape[1]):#predict.shape[1]
            v=predict[i-1:i, j-1:j]
            if v>=0.5:
                XX[i-1:i, j-1:j]=1
            else:
                XX[i-1:i, j-1:j]=0
################################################
scores = loaded_model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

# Compute ROC curve and ROC area for each class ver los algoritmos de Bowles [18]. 
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], XX[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), XX.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#####################################################
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], XX[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), XX.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#####################################################
lw = 2
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue',
                'red','orange','blue','yellow','darkblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()











