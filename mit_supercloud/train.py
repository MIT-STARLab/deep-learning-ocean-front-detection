import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from models import CEDN,HED_small,CEDN_small,HED
import os
import h5py
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import tensorflow.keras.backend as K

def my_loss(y_true,y_pred):
    def sigmoid_focal_crossentropy(y_true,y_pred,alpha = 0.25,gamma = 2.0):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        ce = K.binary_crossentropy(y_true, y_pred, from_logits=False)
        pred_prob = y_pred
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha_factor = 1.0
        modulating_factor = 1.0
        if alpha:
            alpha = tf.cast(alpha, dtype=y_true.dtype)
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        if gamma:
            gamma = tf.cast(gamma, dtype=y_true.dtype)
            modulating_factor = tf.pow((1.0 - p_t), gamma)
        return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
    return sigmoid_focal_crossentropy

#things that change each run
NAME_OF_MODEL = 'cedn_small.h5'
ARCHITECTURE = CEDN_small()

#load model architecture, compile
model = ARCHITECTURE
opt = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(loss=my_loss(), optimizer=opt, metrics=['accuracy'])

#load boa training data
X = np.load('data/boa_training_input.npy')
Y = np.load('data/boa_training_output.npy')

#train model
save_model = ModelCheckpoint(NAME_OF_MODEL, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)
stop_training = EarlyStopping(monitor='val_accuracy', patience=10)
model.fit(X, Y, validation_split=0.1, epochs=100, batch_size=10, callbacks=[save_model,stop_training])