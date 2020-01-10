import sys
import scipy
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers


#  load data
data=np.loadtxt('data/FB15k-237/multilabel/multilabel_vec_(bat_mlp(2,2)).txt')
# data, meta = scipy.io.arff.loadarff('yeast-train.arff')
df=pd.DataFrame(data)

X = df.iloc[:,:200].values
y = df.iloc[:,200:].values
# X = df.iloc[:,0:103].values
# y = df.iloc[:,103:117].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)

# 两层神经网络，每层神经元个数为500，100
def deep_model(feature_dim,label_dim):
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))
    # model.add(Dense(500, activation='relu', input_dim=feature_dim))
    model.add(Dense(100, input_dim=feature_dim))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))

    # model.add(Dense(100, activation='relu'))
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(0.5))

    model.add(Dense(label_dim, activation='sigmoid'))

    adam=optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()

def train_deep(X_train,y_train,X_test,y_test):
    feature_dim = X_train.shape[1]
    label_dim = y_train.shape[1]
    model = deep_model(feature_dim,label_dim)
    model.summary()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
    hist=model.fit(X_train,y_train,batch_size=10, epochs=5,validation_data=(X_test,y_test))
    print(hist.history)
    plt.subplot(122)
    plt.plot(hist.history['loss'])
    plt.show()
    training_vis(hist)
train_deep(X_train,y_train,X_test,y_test)