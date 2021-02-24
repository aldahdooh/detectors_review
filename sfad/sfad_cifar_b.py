from common.util import *

import keras
from keras import backend as K
from keras import optimizers
from keras import regularizers

from sfad.sfad_cifar_a import sfad_a as model_cifar10_adv

class sfad_b:
    def __init__(self, mode='train', no_defense_h5="cifar10_model_1.h5", filename_a="multi_cifar10_model_v7.h5", filename_b="multi_cifar10_model_v7b.h5", coverage=0.95, coverage_th=0.5, alpha=0.25, normalize_mean=False):
        self.mode = mode
        self.no_defense_h5 = no_defense_h5
        self.filename_b = filename_b
        self.filename_a = filename_a
        self.coverage = coverage
        self.coverage_th = coverage_th
        self.alpha = alpha
        self.normalize_mean = False
        self.num_classes = 10
        self.lamda = 32

        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cifar10_data()
        if normalize_mean:
            self.x_train, self.x_test = normalize_mean(self.x_train, self.x_test)
        else: # linear 0-1
            self.x_train, self.x_test = normalize_linear(self.x_train, self.x_test)

        #convert labels to one_hot
        self.y_test_labels = self.y_test[:,0]
        self.y_train, self.y_test = toCat_onehot(self.y_train, self.y_test, self.num_classes+1)

        self.input_shape = self.x_train.shape[1:]

        ### to get prob for each model
        self.model_class = model_cifar10_adv(mode='load', no_defense_h5=self.no_defense_h5, filename=self.filename_a, 
            coverage=coverage, alpha=alpha, normalize_mean=normalize_mean)
        self.model_1 = self.model_class.model_1
        self.model_2 = self.model_class.model_2
        self.model_3 = self.model_class.model_3


        #self.model_a = self.model_class.model
        c = self.model_class.coverage
        l = self.model_class.lamda
        learning_rate = 0.01
        lr_decay = 1e-6

        def selective_loss(y_true, y_pred):
            loss = K.categorical_crossentropy(
                K.repeat_elements(
                    y_pred[:, -1:], self.model_class.num_classes, axis=1) * y_true[:, :-1],
                y_pred[:, :-1]) + l * K.maximum(-K.mean(y_pred[:, -1]) + c, 0) ** 2
            return loss

        def selective_acc(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            temp1 = K.sum(
                (g) * K.cast(K.equal(K.argmax(y_true[:, :-1], axis=-1), K.argmax(y_pred[:, :-1], axis=-1)), K.floatx()))
            temp1 = temp1 / K.sum(g)
            return K.cast(temp1, K.floatx())

        def coverage(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            return K.mean(g)

        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.model_1.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
                      loss_weights=[self.alpha, 1 - self.alpha],
                      optimizer=sgd, metrics=['accuracy', selective_acc])
        self.model_2.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
                      loss_weights=[self.alpha, 1 - self.alpha],
                      optimizer=sgd, metrics=['accuracy', selective_acc])
        self.model_3.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
                      loss_weights=[self.alpha, 1 - self.alpha],
                      optimizer=sgd, metrics=['accuracy', selective_acc])
        
        self.predections_a_train = self.model_1.predict(self.model_class.l_1_train)
        self.predections_b_train = self.model_2.predict(self.model_class.l_2_train)
        self.predections_c_train = self.model_3.predict(self.model_class.l_3_train)
        self.x_train = np.concatenate((self.predections_a_train[0][:, :-1], self.predections_b_train[0][:, :-1], self.predections_c_train[0][:, :-1]), axis=1)
        self.input_shape = self.x_train.shape[1:]

        self.predections_a_test = self.model_1.predict(self.model_class.l_1_test)
        self.predections_b_test = self.model_2.predict(self.model_class.l_2_test)
        self.predections_c_test = self.model_3.predict(self.model_class.l_3_test)
        self.x_test = np.concatenate((self.predections_a_test[0][:, :-1], self.predections_b_test[0][:, :-1], self.predections_c_test[0][:, :-1]), axis=1)
        
        self.model = self.build_model()

        if mode=='train':
            self.model = self.train(self.model)
        elif mode=='load':
            self.model.load_weights("results/sfad/{}".format(self.filename_b))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")
    

    def build_model(self):
        weight_decay = 0.0005
        basic_dropout_rate = 0.1
        input = Input(shape=self.input_shape)

        task0 = Dense(128, kernel_regularizer=regularizers.l2(weight_decay))(input)
        task0 = BatchNormalization()(task0)
        task0 = Activation('relu')(task0)
        # task0 = Dense(self.num_classes, use_bias=True, activation='softmax')(task0)
        # task0 = RBFLayer(128, betas=2)(task0)
        # model_output = Dense(self.num_classes, use_bias=False, activation='softmax')(task0)
        model_output = Dense(self.num_classes, name='model_before_softmax')(task0)
        model_output = Activation('softmax', name='model_after_softmax')(model_output)

        # selection head (g)
        g = Dense(128, kernel_regularizer=regularizers.l2(weight_decay))(task0)
        g = Activation('relu')(g)
        g = BatchNormalization()(g)
        g = Lambda(lambda x: x / 10)(g)
        g = Dense(1, activation='sigmoid', name='g_before_sigmoid')(g)
        g = Activation('sigmoid', name='g_after_sigmoid')(g)
        selective_output = Concatenate(axis=1, name="selective_output")([model_output, g])

        # auxiliary head (h)
        auxiliary_output = Dense(self.num_classes, activation='softmax', name='aux')(task0)
        #auxiliary_output = Activation('softmax', name='aux_after_softmax')(auxiliary_output)

        model = Model(inputs=input, outputs=[selective_output, auxiliary_output])
        return model

    def train(self, model):
        c = self.coverage
        lamda = self.lamda

        def selective_loss(y_true, y_pred):
            loss = K.categorical_crossentropy(
                K.repeat_elements(y_pred[:, -1:], self.num_classes, axis=1) * y_true[:, :-1],
                y_pred[:, :-1]) + lamda * K.maximum(-K.mean(y_pred[:, -1]) + c, 0) ** 2
            return loss

        def selective_acc(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], self.coverage_th), K.floatx())
            temp1 = K.sum(
                (g) * K.cast(K.equal(K.argmax(y_true[:, :-1], axis=-1), K.argmax(y_pred[:, :-1], axis=-1)), K.floatx()))
            temp1 = temp1 / K.sum(g)
            return K.cast(temp1, K.floatx())

        def coverage(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], self.coverage_th), K.floatx())
            return K.mean(g)
        
        batch_size = 200
        maxepoches = 30
        learning_rate = 0.01
        lr_decay = 1e-6
        lr_drop = 10
        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
            optimizer=sgd, metrics=['accuracy', selective_acc], loss_weights=[0.5, 0.5])

        historytemp = model.fit(self.x_train, [self.y_train, self.y_train[:, :-1]], batch_size=batch_size, epochs=maxepoches, callbacks=[reduce_lr],
                                            validation_data=(self.x_test, [self.y_test, self.y_test[:, :-1]]))
        
        with open("results/sfad/{}_history.pkl".format(self.filename_b[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model.save_weights("results/sfad/{}".format(self.filename_b))

        return model
