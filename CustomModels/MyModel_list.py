import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import kld

from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout

from sklearn.model_selection import train_test_split
import time

class StockPred:
    def __init__(self,
                 X,
                 y,
                 epochs,
                 batch_size,
                 ticker,
                 layers,
                 features,
                 neurons,
                 use_reg,
                 test_ratio=.1,
                 val_ratio=.1):
        self.X = X
        self.y = y
        self.epochs = epochs
        self.batch_size = batch_size
        self.ticker = ticker
        self.layers = layers
        self.features = features
        self.neurons = neurons
        self.use_reg = use_reg

        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        self.reset()
        self.gen_split()
        self.gen_cb()

    def gen_layers(self, n, neurons, dropout, norm, inputs):
        x = inputs
        for i in range(n):
            n_neu = neurons // (i + 1)
            if self.use_reg:
                x = Dense(n_neu,name='Dense_'+str(n_neu), activation='elu', kernel_regularizer=regularizers.l1_l2(1e-6,1e-6))(x)
            else:
                x = Dense(n_neu,name='Dense_'+str(n_neu), activation='elu')(x)
            if dropout:
                x = Dropout(dropout,name='Dropout_'+str(dropout)+'_'+str(i))(x)
            if norm:
                x = BatchNormalization(name='Normalization_'+str(i))(x)

        return x

    def gen_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_ratio, shuffle=False)
        self.X_train, self.X_test = X_train.reshape((-1, self.features)), X_test.reshape((-1, self.features))
        self.y_train, self.y_test = y_train.ravel(), y_test.ravel()

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=self.test_ratio, shuffle=False)
        # self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=self.val_ratio, shuffle=False)

    def reset(self):
        K.clear_session()
        name = time.strftime('%d.%m-%H:%M_', time.localtime())
        self.name = name + f'{self.ticker}-' \
                           f'{self.epochs}E' \
                           f'{self.layers}L' \
                           f'{self.features}F' \
                           f'{self.batch_size}B' \
                           f'{self.neurons}N'

    def gen_cb(self):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=0.000001,verbose=1)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose=0, mode='auto')
        self.callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs/' + self.name), reduce_lr,early_stop]

    def gen_model(self, use_dropout, use_norm):
        self.reset()

        inputs = Input(shape=(self.features,),name='Input')
        x = inputs

        x = self.gen_layers(self.layers, self.neurons, use_dropout, use_norm, x)
        pred = Dense(1, activation='tanh',name='Output')(x)

        self.model = Model(inputs=inputs, outputs=pred)

    def compile(self, verbose=0):
        self.model.compile(optimizer='adam',
                           metrics=[tf.keras.metrics.mse],
                           loss=self.jsd)

        self.model.fit(self.X_train,
                       self.y_train,
                       shuffle = True,
                       epochs=self.epochs,
                       verbose=verbose,
                       callbacks = self.callbacks,
                       validation_split=self.val_ratio)


        print(f'Model {self.name} compiled! Calculating accuracy...')
        self.acc = 0  # self.test_model()
        # print(f'Test accuracy is {self.acc}')

        return self.acc

    def jsd(self, y_true, y_pred):
        # K-L divergence and Jensen-Shannon divergence, presumably J-S is better
        # wiki explaines this https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

        y_mean = 0.5 * (y_true + y_pred)
        return 0.5 * kld(y_true, y_mean) + 0.5 * kld(y_pred, y_mean)

    def test_model(self):
        self.prediction = self.model.predict(self.X_test)
        return self.prediction

    def check(self):
        print('this is a test', self.epochs)
