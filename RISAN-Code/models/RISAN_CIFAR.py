from __future__ import print_function

import keras
import numpy as np
import os
import pickle
from keras import backend as K
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate, concatenate
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.core import Lambda
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from RISAN_utils import *
from keras.engine.topology import Layer
import numpy as np
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

class CIFARVgg:
    def __init__(self, train=True, filename="weightsvggnt.h5",alpha=0.7, gamma = 1e-3,maxepochs = 250,cost_rej=0.5,noise_frac=0.0):
        self.alpha = alpha
        self.num_classes = 2
        self.weight_decay= 1e-4
        self.weight_decay_fc= 1e-7
        self.weight_decay_rc= 1e-7
        self.noise = noise_frac
        self._load_data()
        self.gamma = gamma
        self.d = cost_rej
        self.x_shape = self.x_train.shape[1:]
        self.filename = filename
        self.maxepochs = maxepochs

        self.model = self.build_model()

        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights("history_checkpoints/{}".format(self.filename))

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        acti = 'relu'
        # weight_decay = self.weight_decay
        weight_decay = self.weight_decay
        weight_decay_fc = self.weight_decay_fc
        weight_decay_rc = self.weight_decay_rc
        basic_dropout_rate = 0.3
        inputa = Input(shape=self.x_shape)
        curr = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block1_conv1',trainable=True)(inputa)

        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate)(curr)

        curr = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block1_conv2',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2),name = 'block1_pool',trainable=True)(curr)

        curr = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block2_conv1',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block2_conv2',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2),name = 'block2_pool',trainable=True)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block3_conv1',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block3_conv2',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block3_conv3',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2),name = 'block3_pool',trainable=True)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block4_conv1',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block4_conv2',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block4_conv3',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2),name = 'block4_pool',trainable=True)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block5_conv1',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block5_conv2',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),activation=acti,name = 'block5_conv3',trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2),name = 'block5_pool',trainable=True)(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)

        curr_all = Flatten()(curr)
        curr = Dense(512, kernel_regularizer=regularizers.l2(weight_decay_fc),activation=acti,trainable=True)(curr_all)

        curr = BatchNormalization()(curr)
        # curr_a = Dropout(basic_dropout_rate + 0.3)(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)

        curr = Dense(256, kernel_regularizer=regularizers.l2(weight_decay_fc),activation=acti,trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)

        curr = Dense(128, kernel_regularizer=regularizers.l2(weight_decay_fc),activation=acti,trainable=True)(curr)

        curr = BatchNormalization()(curr)
        curr_aux = Dropout(basic_dropout_rate + 0.2)(curr)

        # prediction head (f)
        curr_pred = Dense(1,kernel_regularizer=regularizers.l2(weight_decay_fc))(curr_aux)

        curr2 = Dense(512, kernel_regularizer=regularizers.l2(weight_decay_rc),activation=acti,trainable=True)(curr_all)

        curr2 = BatchNormalization()(curr2)
        curr2 = Dropout(basic_dropout_rate + 0.2)(curr2)

        curr2 = Dense(256, kernel_regularizer=regularizers.l2(weight_decay_rc),activation=acti,trainable=True)(curr2)

        curr2 = BatchNormalization()(curr2)
        curr2 = Dropout(basic_dropout_rate + 0.2)(curr2)

        curr2 = Dense(64, kernel_regularizer=regularizers.l2(weight_decay_rc),activation=acti,trainable=True)(curr2)

        curr2 = BatchNormalization()(curr2)
        curr2 = Dropout(basic_dropout_rate + 0.2)(curr2)

        curr_rho = Dense(1,activation='relu',kernel_regularizer=regularizers.l2(weight_decay_rc),trainable=True)(curr2)


        DEAN_output = concatenate([curr_pred, curr_rho])
        # auxiliary head (h)
        auxiliary_output = Dense(2, activation='softmax', name="classification_head")(curr_aux)

        self.model = Model(inputs=[inputa], outputs=[DEAN_output,auxiliary_output])

        return self.model

    def normalize(self, X_train, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.

        mean = np.mean(X_train, axis=(0, 1, 2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def binary_cifar(self,x_train,y_train):
        # Select samples corresponding to class "Automobile" and "Trucks"
        x_train_n = []
        y_train_n = []
        for a,b in zip(x_train,y_train):
            if b == 9:
                x_train_n.append(a)
                y_train_n.append(0)
            elif b == 1:
                x_train_n.append(a)
                y_train_n.append(1)

        y_train = np.asarray(y_train_n).reshape((len(y_train_n),1))
        x_train = np.asarray(x_train_n)

        return x_train,y_train
    def _load_data(self):

        # The data, shuffled and split between train and test sets:

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        (x_train,y_train) = self.binary_cifar(x_train,y_train)
        (x_test,y_test_label) = self.binary_cifar(x_test,y_test)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        self.x_train, self.x_test = self.normalize(x_train, x_test)
        np.random.seed(1)
        x = np.random.randint(0,len(y_train),int(self.noise*len(y_train)))
        for ind in x:
             if y_train[ind] == 0:
                  y_train[ind] = 1
             elif y_train[ind] == 1:
                  y_train[ind] = 0 
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test_label, self.num_classes)

    def train(self, model):


        def double_sigmoid_loss(y_true, y_pred):

            y_true = y_true[:,1]
            gamma = 1e-3
            y_true = tf.where(y_true<1.0,-1.0,y_true)
            loss = 2*self.d*tf.math.sigmoid(-self.gamma*(y_pred[:,0]*y_true-y_pred[:,1])) + 2*(1-self.d)*tf.math.sigmoid(-self.gamma*(y_true*y_pred[:,0]+y_pred[:,1]))

            return loss

        def double_sigmoid_accuracy(y_true,y_pred):
            y_pred = K.cast(y_pred,K.floatx())
            tn0 = tf.reduce_sum(tf.cast(tf.math.logical_or(tf.math.greater(y_pred[:,0],y_pred[:,1]),tf.math.less_equal(y_pred[:,0],y_pred[:,1])),tf.float32))
            tot = tf.reduce_sum(tf.cast(tf.math.logical_or(tf.math.greater(y_pred[:,0],y_pred[:,1]),tf.math.less(y_pred[:,0],-y_pred[:,1])),tf.float32))
            tn1 = tf.cast(tf.math.logical_or(tf.math.greater(y_pred[:,0],y_pred[:,1]),tf.math.less(y_pred[:,0],-y_pred[:,1])),tf.float32)
            r1 = tf.cast(tf.multiply(y_pred,tf.reshape(tn1,(-1,1))),tf.float32)
            r2 = tf.where(tn1<1.0,-1.0,tn1)
            r3 = tf.cast(tf.math.greater(r1[:,0],r1[:,1]),tf.float32)
            r4 = tf.where(tf.less_equal(r2,r3),r2,r3)
            r5 = tf.reduce_sum(tf.cast(tf.equal(r4,y_true[:,1]),tf.float32))
            r6 = tf.reduce_sum(tf.cast(tf.equal(r4,-1.0),tf.float32))

            acc = r5*(tn0-r6)**-1

            return tf.cast(acc,tf.float32)
        def rho_acc(y_true,y_pred):
            y_pred = K.cast(y_pred,K.floatx())
            tn0 = tf.reduce_sum(tf.cast(tf.math.logical_or(tf.math.greater(y_pred[:,0],y_pred[:,1]),tf.math.less_equal(y_pred[:,0],y_pred[:,1])),tf.float32))
            tn1 = tf.cast(tf.math.logical_or(tf.math.greater(y_pred[:,0],y_pred[:,1]),tf.math.less(y_pred[:,0],-y_pred[:,1])),tf.float32)
            r1 = tf.cast(tf.multiply(y_pred,tf.reshape(tn1,(-1,1))),tf.float32)
            r2 = tf.where(tn1<1.0,-1.0,tn1)
            r3 = tf.cast(tf.math.greater(r1[:,0],r1[:,1]),tf.float32)
            r4 = tf.where(tf.less_equal(r2,r3),r2,r3)
            r5 = tf.reduce_sum(tf.cast(tf.equal(r4,y_true[:,1]),tf.float32))
            r6 = tf.reduce_sum(tf.cast(tf.equal(r4,-1.0),tf.float32))

            acc = r6/tn0
            return tf.cast(acc,tf.float32)

        def coverage(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            return K.mean(g)

        # training parameters
        batch_size = 64
        maxepoches = self.maxepochs
        learning_rate = 1e-1

        lr_decay = 1e-6

        lr_drop = 25

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        #reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=20,min_lr=0.00001,min_delta=0.0001,verbose=1)
        # es = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=25,min_delta=0.001)
        # data augmentation
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_concatenate_1_loss',factor=0.5,patience=20,min_lr=0.00001,min_delta=0.0001,verbose=1)
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(self.x_train)
        ep = 1e-07
        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(lr=learning_rate,epsilon=ep)
        model.compile(loss=[double_sigmoid_loss,'categorical_crossentropy'],loss_weights=[self.alpha,1-self.alpha],
                      optimizer=sgd, metrics=['accuracy', double_sigmoid_accuracy,rho_acc])

        historytemp = model.fit_generator(my_generator(datagen.flow,self.x_train,self.y_train,batch_size,2),
                                          steps_per_epoch=self.x_train.shape[0] // batch_size,
                                          epochs=maxepoches, callbacks=[reduce_lr],
                                        initial_epoch=0,
                                        validation_data=([self.x_test], [self.y_test,self.y_test]))

        with open("history_checkpoints/{}_history.pkl".format(self.filename[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # model.save_weights("history_checkpoints/{}".format(self.filename))

        return model

