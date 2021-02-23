from __future__ import division, absolute_import, print_function
from common.util import *
from setup_paths import *

class SVHNCNN:
    def __init__(self, mode='train', filename="cnn_svhn.h5", normalize_mean=False, epochs=50, batch_size=128):
        self.mode = mode #train or load
        self.filename = filename
        self.normalize_mean = normalize_mean
        self.epochs = epochs
        self.batch_size = batch_size

        #====================== load data ========================
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_svhn_data()
        if normalize_mean:
            self.x_train, self.x_test = normalize_mean(self.x_train, self.x_test)
        else: # linear 0-1
            self.x_train, self.x_test = normalize_linear(self.x_train, self.x_test)

        #convert labels to one_hot
        self.y_test_labels = self.y_test
        self.y_train_labels = self.y_train
        self.y_train, self.y_test = toCat_onehot(self.y_train, self.y_test, self.num_classes)

        #====================== Model =============================
        self.input_shape = self.x_train.shape[1:]
        self.model = self.build_model()

        if mode=='train':
            self.model = self.train(self.model)
        elif mode=='load':
            self.model.load_weights("{}{}{}".format(checkpoints_dir, 'gray/', self.filename))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")

    def build_model(self):
        #================= Settings =========================
        weight_decay = 0.0005
        basic_dropout_rate = 0.3

        #================= Input ============================
        input = Input(shape=self.input_shape, name='l_0')

        #================= CONV ============================
        task0 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_1')(input)
        task0 = Activation('relu', name='l_2')(task0)

        task0 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_3')(task0)
        task0 = Activation('relu', name='l_4')(task0)
        task0 = MaxPooling2D(pool_size=(2, 2), name='l_5')(task0)

        task0 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_6')(task0)
        task0 = Activation('relu', name='l_7')(task0)

        task0 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_8')(task0)
        task0 = Activation('relu', name='l_9')(task0)
        task0 = MaxPooling2D(pool_size=(2, 2), name='l_10')(task0) 

        #================= Dense ============================
        task0 = Flatten(name='l_11')(task0)
        task0 = Dense(512, kernel_regularizer=l2(weight_decay), name='l_12')(task0)
        task0 = Activation('relu', name='l_13')(task0)
        task0 = Dropout(basic_dropout_rate + 0.2, name='l_14')(task0) 
        task0 = Dense(128, kernel_regularizer=l2(weight_decay), name='l_15')(task0)
        task0 = Activation('relu', name='l_16')(task0) 
        task0 = Dropout(basic_dropout_rate + 0.2, name='l_17')(task0) 
        
        #================= Output - classification head ============================
        classification_output = Dense(self.num_classes, name="classification_head_before_activation")(task0)
        classification_output = Activation('softmax', name="classification_head")(classification_output)

        #================= The final model ============================
        model = Model(inputs=input, outputs=classification_output)
        return model
    
    def train(self, model):
        #================= Settings =========================
        learning_rate = 0.01
        lr_decay = 1e-6
        lr_drop = 50
        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = LearningRateScheduler(lr_scheduler, verbose=1)
        weights_file = "{}{}{}".format(checkpoints_dir, "gray/", self.filename)
        model_checkpoint = ModelCheckpoint(weights_file, monitor='val_accuracy', save_best_only=True, verbose=1)
        callbacks=[reduce_lr, model_checkpoint]

        #================= Data augmentation =========================
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
        
        #================= Train =========================
        datagen.fit(self.x_train)
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

        historytemp = model.fit_generator(datagen.flow(self.x_train, y=self.y_train, batch_size=self.batch_size),
                                          epochs=self.epochs, callbacks=callbacks,
                                          validation_data=(self.x_test, self.y_test))
        
        #================= Save model and history =========================
        with open("{}{}{}_history.pkl".format(checkpoints_dir, "gray/", self.filename[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # model.save_weights(weights_file)

        return model
