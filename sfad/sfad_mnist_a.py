from common.util import *

import keras
from keras import optimizers
from keras import regularizers

class sfad_a:
    def __init__(self, mode='train', no_defense_h5="mnist_model_1.h5", filename="multi_mnist_model_v7.h5", coverage=0.95, coverage_th=0.5, alpha=0.5 , normalize_mean=False):
        self.mode = mode
        self.filename = filename
        self.coverage = coverage
        self.coverage_th = coverage_th
        self.normalize_mean = normalize_mean
        self.alpha = alpha
        self.num_classes = 10
        self.lamda = 32

        ## clean data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_mnist_data()
        if normalize_mean:
            self.x_train, self.x_test = normalize_mean(self.x_train, self.x_test)
        else: # linear 0-1
            self.x_train, self.x_test = normalize_linear(self.x_train, self.x_test)

        #convert labels to one_hot
        self.y_test_labels = self.y_test
        self.y_train, self.y_test = toCat_onehot(self.y_train, self.y_test, self.num_classes + 1)

        self.input_shape = self.x_train.shape[1:]

        from baselineCNN.cnn.cnn_mnist import MNISTCNN as no_defense
        model_nodefense_class = no_defense(mode="load", filename=no_defense_h5)
        self.no_defense_model = model_nodefense_class.model
        learning_rate = 0.1
        lr_decay = 1e-6
        
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.no_defense_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        
        self.no_defense_model.summary()
        loss_test, class_head_acc = self.no_defense_model.evaluate(model_nodefense_class.x_test, model_nodefense_class.y_test)
        print('Loss::{:4.4f} and Accuracy::{:4.2f}%  on test data'.format(loss_test, class_head_acc * 100))

        ##layers data train
        inter_model = Model(inputs=self.no_defense_model.get_input_at(0), outputs=self.no_defense_model.get_layer('l_16').output)
        self.l_1_train = inter_model.predict(self.x_train)
        if len(self.l_1_train.shape)==2:
            self.l_1_train = self.l_1_train.reshape(self.l_1_train.shape[0],1, 1, self.l_1_train.shape[1])
        elif len(self.l_1_train.shape)==4:
            self.l_1_train = self.l_1_train.reshape(self.l_1_train.shape[0],self.l_1_train.shape[1], self.l_1_train.shape[2], self.l_1_train.shape[3])
        self.l_1_input_shape = self.l_1_train.shape[1:]

        inter_model = Model(inputs=self.no_defense_model.get_input_at(0), outputs=self.no_defense_model.get_layer('l_14').output)
        self.l_2_train = inter_model.predict(self.x_train)
        if len(self.l_2_train.shape)==2:
            self.l_2_train = self.l_2_train.reshape(self.l_2_train.shape[0],1, 1, self.l_2_train.shape[1])
        elif len(self.l_2_train.shape)==4:
            self.l_2_train = self.l_2_train.reshape(self.l_2_train.shape[0],self.l_2_train.shape[1], self.l_2_train.shape[2], self.l_2_train.shape[3])
        self.l_2_input_shape = self.l_2_train.shape[1:]

        inter_model = Model(inputs=self.no_defense_model.get_input_at(0), outputs=self.no_defense_model.get_layer('l_10').output)
        self.l_3_train = inter_model.predict(self.x_train)
        if len(self.l_3_train.shape)==2:
            self.l_3_train = self.l_3_train.reshape(self.l_3_train.shape[0],1, 1, self.l_3_train.shape[1])
        elif len(self.l_3_train.shape)==4:
            self.l_3_train = self.l_3_train.reshape(self.l_3_train.shape[0],self.l_3_train.shape[1], self.l_3_train.shape[2], self.l_3_train.shape[3])
        self.l_3_input_shape = self.l_3_train.shape[1:]

        ##layers data test
        inter_model = Model(inputs=self.no_defense_model.get_input_at(0), outputs=self.no_defense_model.get_layer('l_16').output)
        self.l_1_test = inter_model.predict(self.x_test)
        if len(self.l_1_test.shape)==2:
            self.l_1_test = self.l_1_test.reshape(self.l_1_test.shape[0],1, 1, self.l_1_test.shape[1])
        elif len(self.l_1_test.shape)==4:
            self.l_1_test = self.l_1_test.reshape(self.l_1_test.shape[0],self.l_1_test.shape[1], self.l_1_test.shape[2], self.l_1_test.shape[3])

        inter_model = Model(inputs=self.no_defense_model.get_input_at(0), outputs=self.no_defense_model.get_layer('l_14').output)
        self.l_2_test = inter_model.predict(self.x_test)
        if len(self.l_2_test.shape)==2:
            self.l_2_test = self.l_2_test.reshape(self.l_2_test.shape[0],1, 1, self.l_2_test.shape[1])
        elif len(self.l_2_test.shape)==4:
            self.l_2_test = self.l_2_test.reshape(self.l_2_test.shape[0],self.l_2_test.shape[1], self.l_2_test.shape[2], self.l_2_test.shape[3])

        inter_model = Model(inputs=self.no_defense_model.get_input_at(0), outputs=self.no_defense_model.get_layer('l_10').output)
        self.l_3_test = inter_model.predict(self.x_test)
        if len(self.l_3_test.shape)==2:
            self.l_3_test = self.l_3_test.reshape(self.l_3_test.shape[0],1, 1, self.l_3_test.shape[1])
        elif len(self.l_3_test.shape)==4:
            self.l_3_test = self.l_3_test.reshape(self.l_3_test.shape[0],self.l_3_test.shape[1], self.l_3_test.shape[2], self.l_3_test.shape[3])

        ## build the model
        self.model_1, self.model_2, self.model_3 = self.build_model()

        if mode=='train':
            self.model_1, self.model_2, self.model_3 = self.train(self.model_1, self.model_2, self.model_3)
        elif mode=='load':
            self.model_1.load_weights("results/sfad/{}".format(self.filename[:-3]+'_model_1.h5'))
            self.model_2.load_weights("results/sfad/{}".format(self.filename[:-3]+'_model_2.h5'))
            self.model_3.load_weights("results/sfad/{}".format(self.filename[:-3]+'_model_3.h5'))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")
    

    def build_model(self):
        weight_decay = 0.0005
        basic_dropout_rate = 0.1

        #inputs
        inputa = Input(shape=self.l_1_input_shape)
        inputb = Input(shape=self.l_2_input_shape)
        inputc = Input(shape=self.l_3_input_shape)

        ###########################   for model_1
        ######## for Clean features
        #1a.encode    
        taska1 = Conv2D(np.int32(np.int32(self.l_1_input_shape[2]/2)), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputa)
        taska1 = BatchNormalization()(taska1)
        taska1 = Activation('relu')(taska1)
        taska1 = Conv2D(np.int32(self.l_1_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taska1)
        taska1 = BatchNormalization()(taska1)
        taska1 = Activation('relu')(taska1)
        taska1 = Conv2D(np.int32(self.l_1_input_shape[2]/16), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taska1)
        taska1 = BatchNormalization()(taska1)
        taska1 = Activation('relu')(taska1)
        #1b.decode
        taska1 = Conv2D(np.int32(self.l_1_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taska1)
        taska1 = BatchNormalization()(taska1)
        taska1 = Activation('relu')(taska1)
        taska1 = Conv2D(np.int32(self.l_1_input_shape[2]/2), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taska1)
        taska1 = BatchNormalization()(taska1)
        taska1 = Activation('relu')(taska1)
        taska1 = Conv2D(np.int32(self.l_1_input_shape[2]), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taska1)
        #1c.merge
        taska1 = Lambda(lambda inputs: K.abs(inputs[0] + inputs[1]))([inputa, taska1])

        #Upsampling and downsampling
        taska2 = UpSampling2D(size=2, interpolation='bilinear')(taska1)
        taska2 = UpSampling2D(size=2, interpolation='bilinear')(taska2)
        taska2 = AveragePooling2D(pool_size=2, strides=2)(taska2)
        taska2 = AveragePooling2D(pool_size=2, strides=2)(taska2)
        taska2 = Lambda(lambda inputs: inputs[0] + inputs[1])([taska1, taska2])

        #Merge encode/decode and up/down sampling 
        # taska = Concatenate(axis=3)([taska1, taska2])

        ######## for Noise features
        #1a.encode
        inputna = GaussianNoise(1)(inputa)
        taskna1 = Conv2D(np.int32(np.int32(self.l_1_input_shape[2]/2)), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputna)
        taskna1 = BatchNormalization()(taskna1)
        taskna1 = Activation('relu')(taskna1)
        taskna1 = Conv2D(np.int32(self.l_1_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskna1)
        taskna1 = BatchNormalization()(taskna1)
        taskna1 = Activation('relu')(taskna1)
        taskna1 = Conv2D(np.int32(self.l_1_input_shape[2]/16), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskna1)
        taskna1 = BatchNormalization()(taskna1)
        taskna1 = Activation('relu')(taskna1)
        #1b.decode
        taskna1 = Conv2D(np.int32(self.l_1_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskna1)
        taskna1 = BatchNormalization()(taskna1)
        taskna1 = Activation('relu')(taskna1)
        taskna1 = Conv2D(np.int32(self.l_1_input_shape[2]/2), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskna1)
        taskna1 = BatchNormalization()(taskna1)
        taskna1 = Activation('relu')(taskna1)
        taskna1 = Conv2D(np.int32(self.l_1_input_shape[2]), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskna1)
        #1c.merge
        taskna1 = Lambda(lambda inputs: K.abs(inputs[0] + inputs[1]))([inputna, taskna1])
        taskna1 = GaussianNoise(1)(taskna1)

        #Upsampling and downsampling
        taskna2 = UpSampling2D(size=2, interpolation='bilinear')(taskna1)
        taskna2 = UpSampling2D(size=2, interpolation='bilinear')(taskna2)
        taskna2 = AveragePooling2D(pool_size=2, strides=2)(taskna2)
        taskna2 = AveragePooling2D(pool_size=2, strides=2)(taskna2)
        taskna2 = Lambda(lambda inputs: inputs[0] + inputs[1])([taskna1, taskna2])

        #Merge encode/decode and up/down sampling 
        # taskna = Concatenate(axis=3)([taskna1, taskna2])

        #Merge clean and noise
        taska = Concatenate(axis=3)([taska2, taskna2])
        # taska = GaussianNoise(1)(taska)

        #bottelneck block
        taska = Conv2D(1024, (1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taska)
        taska = BatchNormalization()(taska)
        taska = Activation('relu')(taska)
        taska = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taska)
        taska = BatchNormalization()(taska)
        taska = Activation('relu')(taska)
        taska = Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taska)
        taska = BatchNormalization()(taska)
        taska = Activation('relu')(taska)
        # taska = GaussianDropout(0.2)(taska)
        
        taska = Flatten()(taska)
        taska = Dense(512)(taska)
        taska = BatchNormalization()(taska)
        taska = Activation('relu')(taska)
        # taska = GaussianDropout(0.2)(taska)
        # taska = Dense(self.num_classes, activation='softmax')(taska)
        # taska = RBFLayer(512, betas=2)(taska)
        # model_1_output = Dense(self.num_classes, use_bias=False, activation='softmax')(taska)
        model_1_output = Dense(self.num_classes, name='model_1_before_softmax')(taska)
        model_1_output = Activation('softmax', name='model_1_after_softmax')(model_1_output)

        # selection (g1)
        g1 = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(taska)
        g1 = BatchNormalization()(g1)
        g1 = Activation('relu')(g1)
        g1 = Lambda(lambda x: x / 10)(g1)
        g1 = Dense(1, activation='sigmoid')(g1)
        model_1_selective = Concatenate(axis=1, name="model_1_selective")([model_1_output, g1])
        model_1_auxiliary = Dense(self.num_classes, use_bias=False, activation='softmax', name='aux_1')(taska)
        model_1 = Model(inputs=inputa, outputs=[model_1_selective, model_1_auxiliary])

        ###########################   for model_2
        ######## for Clean features
        #1a.encode    
        taskb1 = Conv2D(np.int32(np.int32(self.l_2_input_shape[2]/2)), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputb)
        taskb1 = BatchNormalization()(taskb1)
        taskb1 = Activation('relu')(taskb1)
        taskb1 = Conv2D(np.int32(self.l_2_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskb1)
        taskb1 = BatchNormalization()(taskb1)
        taskb1 = Activation('relu')(taskb1)
        taskb1 = Conv2D(np.int32(self.l_2_input_shape[2]/16), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskb1)
        taskb1 = BatchNormalization()(taskb1)
        taskb1 = Activation('relu')(taskb1)
        #1b.decode
        taskb1 = Conv2D(np.int32(self.l_2_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskb1)
        taskb1 = BatchNormalization()(taskb1)
        taskb1 = Activation('relu')(taskb1)
        taskb1 = Conv2D(np.int32(self.l_2_input_shape[2]/2), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskb1)
        taskb1 = BatchNormalization()(taskb1)
        taskb1 = Activation('relu')(taskb1)
        taskb1 = Conv2D(np.int32(self.l_2_input_shape[2]), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskb1)
        #1c.merge
        taskb1 = Lambda(lambda inputs: K.abs(inputs[0] + inputs[1]))([inputb, taskb1])

        #Upsampling and downsampling
        taskb2 = UpSampling2D(size=2, interpolation='bilinear')(taskb1)
        taskb2 = UpSampling2D(size=2, interpolation='bilinear')(taskb2)
        taskb2 = AveragePooling2D(pool_size=2, strides=2)(taskb2)
        taskb2 = AveragePooling2D(pool_size=2, strides=2)(taskb2)
        taskb2 = Lambda(lambda inputs: inputs[0] + inputs[1])([taskb1, taskb2])

        #Merge encode/decode and up/down sampling 
        # taskb = Concatenate(axis=3)([taskb1, taskb2])

        ######## for Noise features
        #1a.encode
        inputnb = GaussianNoise(1)(inputb)    
        tasknb1 = Conv2D(np.int32(np.int32(self.l_2_input_shape[2]/2)), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputnb)
        tasknb1 = BatchNormalization()(tasknb1)
        tasknb1 = Activation('relu')(tasknb1)
        tasknb1 = Conv2D(np.int32(self.l_2_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(tasknb1)
        tasknb1 = BatchNormalization()(tasknb1)
        tasknb1 = Activation('relu')(tasknb1)
        tasknb1 = Conv2D(np.int32(self.l_2_input_shape[2]/16), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(tasknb1)
        tasknb1 = BatchNormalization()(tasknb1)
        tasknb1 = Activation('relu')(tasknb1)
        #1b.decode
        tasknb1 = Conv2D(np.int32(self.l_2_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(tasknb1)
        tasknb1 = BatchNormalization()(tasknb1)
        tasknb1 = Activation('relu')(tasknb1)
        tasknb1 = Conv2D(np.int32(self.l_2_input_shape[2]/2), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(tasknb1)
        tasknb1 = BatchNormalization()(tasknb1)
        tasknb1 = Activation('relu')(tasknb1)
        tasknb1 = Conv2D(np.int32(self.l_2_input_shape[2]), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(tasknb1)
        #1c.merge
        tasknb1 = Lambda(lambda inputs: K.abs(inputs[0] + inputs[1]))([inputnb, tasknb1])
        tasknb1 = GaussianNoise(1)(tasknb1)

        #Upsampling and downsampling
        tasknb2 = UpSampling2D(size=2, interpolation='bilinear')(tasknb1)
        tasknb2 = UpSampling2D(size=2, interpolation='bilinear')(tasknb2)
        tasknb2 = AveragePooling2D(pool_size=2, strides=2)(tasknb2)
        tasknb2 = AveragePooling2D(pool_size=2, strides=2)(tasknb2)
        tasknb2 = Lambda(lambda inputs: inputs[0] + inputs[1])([tasknb1, tasknb2])

        #Merge encode/decode and up/down sampling 
        # tasknb = Concatenate(axis=3)([tasknb1, tasknb2])

        #Merge clean and noise 
        taskb = Concatenate(axis=3)([taskb2, tasknb2])
        # taskb = GaussianNoise(1)(taskb)

        #bottelneck block
        taskb = Conv2D(1024, (1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskb)
        taskb = BatchNormalization()(taskb)
        taskb = Activation('relu')(taskb)
        taskb = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskb)
        taskb = BatchNormalization()(taskb)
        taskb = Activation('relu')(taskb)
        taskb = Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskb)
        taskb = BatchNormalization()(taskb)
        taskb = Activation('relu')(taskb)
        # taskb = GaussianDropout(0.2)(taskb)

        taskb = Flatten()(taskb)
        taskb = Dense(512)(taskb)
        taskb = BatchNormalization()(taskb)
        taskb = Activation('relu')(taskb)
        # taskb = GaussianDropout(0.2)(taskb)
        # taskb = Dense(self.num_classes, activation='softmax')(taskb)
        # taskb = RBFLayer(512, betas=2)(taskb)
        # model_2_output = Dense(self.num_classes, use_bias=False, activation='softmax')(taskb)
        
        model_2_output = Dense(self.num_classes, name='model_2_before_softmax')(taskb)
        model_2_output = Activation('softmax', name='model_2_after_softmax')(model_2_output)
        

        # selection (g2)
        g2 = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(taskb)
        g2 = BatchNormalization()(g2)
        g2 = Activation('relu')(g2)
        g2 = Lambda(lambda x: x / 10)(g2)
        g2 = Dense(1, activation='sigmoid')(g2)
        model_2_selective = Concatenate(axis=1, name="model_2_selective")([model_2_output, g2])
        model_2_auxiliary = Dense(self.num_classes, use_bias=False, activation='softmax', name='aux_2')(taskb)
        model_2 = Model(inputs=inputb, outputs=[model_2_selective, model_2_auxiliary])

        ###########################   for model_3
        ######## for Clean features
        #1a.encode    
        taskc1 = Conv2D(np.int32(np.int32(self.l_3_input_shape[2]/2)), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputc)
        taskc1 = BatchNormalization()(taskc1)
        taskc1 = Activation('relu')(taskc1)
        taskc1 = Conv2D(np.int32(self.l_3_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskc1)
        taskc1 = BatchNormalization()(taskc1)
        taskc1 = Activation('relu')(taskc1)
        taskc1 = Conv2D(np.int32(self.l_3_input_shape[2]/16), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskc1)
        taskc1 = BatchNormalization()(taskc1)
        taskc1 = Activation('relu')(taskc1)
        #1b.decode
        taskc1 = Conv2D(np.int32(self.l_3_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskc1)
        taskc1 = BatchNormalization()(taskc1)
        taskc1 = Activation('relu')(taskc1)
        taskc1 = Conv2D(np.int32(self.l_3_input_shape[2]/2), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskc1)
        taskc1 = BatchNormalization()(taskc1)
        taskc1 = Activation('relu')(taskc1)
        taskc1 = Conv2D(np.int32(self.l_3_input_shape[2]), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskc1)
        #1c.merge
        taskc1 = Lambda(lambda inputs: K.abs(inputs[0] + inputs[1]))([inputc, taskc1])

        #Upsampling and downsampling
        taskc2 = UpSampling2D(size=2, interpolation='bilinear')(taskc1)
        taskc2 = UpSampling2D(size=2, interpolation='bilinear')(taskc2)
        taskc2 = AveragePooling2D(pool_size=2, strides=2)(taskc2)
        taskc2 = AveragePooling2D(pool_size=2, strides=2)(taskc2)
        taskc2 = Lambda(lambda inputs: inputs[0] + inputs[1])([taskc1, taskc2])

        #Merge encode/decode and up/down sampling 
        # taskc = Concatenate(axis=3)([taskc1, taskc2])

        ######## for Noise features
        #1a.encode
        inputnc = GaussianNoise(1)(inputc)    
        tasknc1 = Conv2D(np.int32(np.int32(self.l_3_input_shape[2]/2)), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputnc)
        tasknc1 = BatchNormalization()(tasknc1)
        tasknc1 = Activation('relu')(tasknc1)
        tasknc1 = Conv2D(np.int32(self.l_3_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(tasknc1)
        tasknc1 = BatchNormalization()(tasknc1)
        tasknc1 = Activation('relu')(tasknc1)
        tasknc1 = Conv2D(np.int32(self.l_3_input_shape[2]/16), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(tasknc1)
        tasknc1 = BatchNormalization()(tasknc1)
        tasknc1 = Activation('relu')(tasknc1)
        #1b.decode
        tasknc1 = Conv2D(np.int32(self.l_3_input_shape[2]/4), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(tasknc1)
        tasknc1 = BatchNormalization()(tasknc1)
        tasknc1 = Activation('relu')(tasknc1)
        tasknc1 = Conv2D(np.int32(self.l_3_input_shape[2]/2), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(tasknc1)
        tasknc1 = BatchNormalization()(tasknc1)
        tasknc1 = Activation('relu')(tasknc1)
        tasknc1 = Conv2D(np.int32(self.l_3_input_shape[2]), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(tasknc1)
        #1c.merge
        tasknc1 = Lambda(lambda inputs: K.abs(inputs[0] + inputs[1]))([inputnc, tasknc1])
        tasknc1 = GaussianNoise(1)(tasknc1)

        #Upsampling and downsampling
        tasknc2 = UpSampling2D(size=2, interpolation='bilinear')(tasknc1)
        tasknc2 = UpSampling2D(size=2, interpolation='bilinear')(tasknc2)
        tasknc2 = AveragePooling2D(pool_size=2, strides=2)(tasknc2)
        tasknc2 = AveragePooling2D(pool_size=2, strides=2)(tasknc2)
        tasknc2 = Lambda(lambda inputs: inputs[0] + inputs[1])([tasknc1, tasknc2])

        #Merge encode/decode and up/down sampling 
        # tasknc = Concatenate(axis=3)([tasknc1, tasknc2])

        #Merge clean and noise 
        taskc = Concatenate(axis=3)([taskc2, tasknc2])
        # taskc = GaussianNoise(1)(taskc)

        #bottelneck block
        taskc = Conv2D(1024, (1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskc)
        taskc = BatchNormalization()(taskc)
        taskc = Activation('relu')(taskc)
        taskc = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskc)
        taskc = BatchNormalization()(taskc)
        taskc = Activation('relu')(taskc)
        taskc = Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(taskc)
        taskc = BatchNormalization()(taskc)
        taskc = Activation('relu')(taskc)
        # taskc = GaussianDropout(0.2)(taskc)

        taskc = Flatten()(taskc)
        taskc = Dense(512)(taskc)
        taskc = BatchNormalization()(taskc)
        taskc = Activation('relu')(taskc)   
        # taskc = GaussianDropout(0.2)(taskc)
        # taskc = Dense(self.num_classes, activation='softmax')(taskc)
        # taskc = RBFLayer(512, betas=2)(taskc)
        # model_3_output = Dense(self.num_classes, use_bias=False, activation='softmax')(taskc)
        model_3_output = Dense(self.num_classes, name='model_3_before_softmax')(taskc)
        model_3_output = Activation('softmax', name='model_3_after_softmax')(model_3_output)

        # selection (g3)
        g3 = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(taskc)
        g3 = BatchNormalization()(g3)
        g3 = Activation('relu')(g3)
        g3 = Lambda(lambda x: x / 10)(g3)
        g3 = Dense(1, activation='sigmoid')(g3)
        model_3_selective = Concatenate(axis=1, name="model_3_selective")([model_3_output, g3])
        model_3_auxiliary = Dense(self.num_classes, use_bias=False, activation='softmax', name='aux_3')(taskc)
        model_3 = Model(inputs=inputc, outputs=[model_3_selective, model_3_auxiliary])

        return model_1, model_2, model_3
    

    def train(self, model_1, model_2, model_3):
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
        lr_drop = 500
        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

        ####################### train model_1
        model_1.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
                      loss_weights=[self.alpha, 1 - self.alpha],
                      optimizer=sgd, metrics=['accuracy', selective_acc])

        x_1 = self.l_1_train
        y_ = [self.y_train, self.y_train[:,:-1]]#, [self.y_train, self.y_train[:,:-1]], [self.y_train, self.y_train[:,:-1]]]
        x_1_val = self.l_1_test
        y_val = [self.y_test, self.y_test[:, :-1]]#, [self.y_test, self.y_test[:, :-1]], [self.y_test, self.y_test[:, :-1]]]
        historytemp = model_1.fit(x_1, y_, batch_size=batch_size,
                                    epochs=maxepoches, callbacks=[reduce_lr],
                                    validation_data=(x_1_val, y_val))
        
        with open("results/sfad/{}_history_model_1.pkl".format(self.filename[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model_1.save_weights("results/sfad/{}_model_1.h5".format(self.filename[:-3]))


        ####################### train model_2
        model_2.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
                      loss_weights=[self.alpha, 1 - self.alpha],
                      optimizer=sgd, metrics=['accuracy', selective_acc])

        x_2 = self.l_2_train
        x_2_val = self.l_2_test
        historytemp = model_2.fit(x_2, y_, batch_size=batch_size,
                                    epochs=maxepoches, callbacks=[reduce_lr],
                                    validation_data=(x_2_val, y_val))
        
        with open("results/sfad/{}_history_model_2.pkl".format(self.filename[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model_2.save_weights("results/sfad/{}_model_2.h5".format(self.filename[:-3]))

        ####################### train model_3
        # maxepoches = 500
        model_3.compile(loss=[selective_loss, keras.losses.categorical_crossentropy], 
                      loss_weights=[self.alpha, 1 - self.alpha],
                      optimizer=sgd, metrics=['accuracy', selective_acc])

        x_3 = self.l_3_train
        x_3_val = self.l_3_test
        historytemp = model_3.fit(x_3, y_, batch_size=batch_size,
                                    epochs=maxepoches, callbacks=[reduce_lr],
                                    validation_data=(x_3_val, y_val))
        
        with open("results/sfad/{}_history_model_3.pkl".format(self.filename[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model_3.save_weights("results/sfad/{}_model_3.h5".format(self.filename[:-3]))

        return model_1, model_2, model_3
