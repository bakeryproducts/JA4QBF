import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense,LSTM,Input,BatchNormalization,Dropout,CuDNNLSTM,CuDNNGRU
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import time


class StockPred:
    def __init__(self,
                 X,
                 y,
                 mode,
                 epochs,
                 window,
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
        self.mode = mode
        self.window = window
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
        self.gen_timeseries()
        self.gen_cb()

    def gen_lstm_layers(self,n, neurons, dropout, norm, inputs):
        rs = True
        x = inputs

        for i in range(n):
            if i == n - 1:
                rs = False
            if self.mode == 'gpu':
                if self.use_reg:
                    x = CuDNNLSTM(neurons//(i+1),
                                  return_sequences=rs,
                                  recurrent_regularizer=regularizers.l2(l=1e-4),
                                  kernel_regularizer=regularizers.l2(l=1e-4)
                                 )(x)
                else:
                    x = CuDNNLSTM(neurons//(i+1),
                                  return_sequences=rs
                                  )(x)
            elif self.mode == 'cpu':
                if self.use_reg:
                    x = LSTM(neurons//(i+1),
                             return_sequences=rs,
                             recurrent_regularizer=regularizers.l2(l=1e-4),
                             kernel_regularizer=regularizers.l2(l=1e-4)
                             )(x)
                else:
                    x = LSTM(neurons//(i+1),
                             return_sequences=rs
                             )(x)
            else:
                if self.use_reg:
                    x = Dense(neurons // (i + 1))(x)
                else:
                    x = Dense(neurons // (i + 1))(x)

            if dropout:
                x = Dropout(dropout)(x)
            if norm:
                x = BatchNormalization()(x)

        return x

    def gen_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=self.test_ratio, shuffle=False)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=self.val_ratio, shuffle=False)
        


    def gen_timeseries(self):
        self.train_gen = TimeseriesGenerator(self.X_train, self.y_train, length=self.window, batch_size=self.batch_size, shuffle=True)
        self.validation_gen = TimeseriesGenerator(self.X_val, self.y_val, length=self.window, batch_size=self.batch_size)
        self.test_gen = TimeseriesGenerator(self.X_test,self.y_test,length=self.window,batch_size=10)
        self.total_gen = TimeseriesGenerator(self.X,self.y,length=self.window,batch_size=10)#self.y.shape[0])


    def reset(self):
        K.clear_session()
        name = time.strftime('%d.%m-%H:%M_', time.localtime())
        self.name = name + f'{self.ticker}-' \
                           f'{self.epochs}E' \
                           f'{self.layers}L' \
                           f'{self.features}F' \
                           f'{self.window}W' \
                           f'{self.batch_size}B' \
                           f'{self.neurons}N'

    def gen_cb(self):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose=0, mode='auto')

        self.callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs/' + self.name),reduce_lr]#,early_stop]


    def gen_model(self,use_dropout,use_norm):
        self.reset()

        inputs = Input(shape=(self.window, self.features,))
        x = inputs

        x = self.gen_lstm_layers(self.layers, self.neurons, use_dropout, use_norm, x)
        #x = Dense(8, activation='relu')(x)
        # if use_dropout:
        #     x = Dropout(use_dropout)(x)
        # if use_norm:
        #     x = BatchNormalization()(x)
        
        pred = Dense(1, activation='tanh')(x)

        self.model = Model(inputs=inputs, outputs=pred)

    def compile(self,verbose=0):
        self.model.compile(optimizer='adam',
                           metrics=[tf.keras.metrics.mse],
                           loss=tf.keras.losses.mse)

        self.model.fit_generator(generator=self.train_gen,
                                 validation_data=self.validation_gen,
                                 shuffle=True,
                                 epochs=self.epochs,
                                 verbose=verbose,
                                 callbacks=self.callbacks,
                                 use_multiprocessing=False,
                                 workers=8)

        print(f'Model {self.name} compiled! Calculating accuracy...')
        self.acc = 0#self.test_model()
        #print(f'Test accuracy is {self.acc}')

        return self.acc

    def test_model(self):
        scores = self.model.evaluate_generator(self.test_gen, verbose=1)
        self.acc = scores[1]
        return self.acc

    def check(self):
        print('this is a test',self.epochs)