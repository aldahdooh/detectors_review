from __future__ import division, absolute_import, print_function
from common.util import *
from setup_paths import *
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import preprocess_input as pre_resnet
from sklearn.utils import shuffle

class TINYCNN:
    def __init__(self, mode='train', filename="cnn_tiny.h5", epochs=300, batch_size=128):
        self.mode = mode #train or load
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size

        #====================== load data ========================
        self.num_classes = 200
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_tiny_imagenet_data()
        self.x_train, self.x_test = normalize_linear(self.x_train, self.x_test)
        if mode=='train':
            self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        # self.x_train = pre_dense(self.x_train)
        # self.x_test = pre_dense(self.x_test)

        #convert labels to one_hot
        self.y_test_labels = self.y_test
        self.y_train_labels = self.y_train
        self.y_train, self.y_test = toCat_onehot(self.y_train, self.y_test, self.num_classes)

        #====================== Model =============================
        self.input_shape = self.x_train.shape[1:]
        self.model = self.build_model()

        if mode=='train':
            # if os.path.isfile("checkpoints/{}".format(self.filename)):
            #     self.model.load_weights("checkpoints/{}".format(self.filename))
            self.model = self.train(self.model)
        elif mode=='load':
            self.model.load_weights("{}{}{}".format(checkpoints_dir, "gray/", self.filename))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")

    def build_model(self):
        #================= Settings =========================
        weight_decay = 0.001

        #================= Dense ============================
        base_model = ResNet50V2(weights='imagenet', input_shape=(self.input_shape), include_top=False)
        # base_model = ResNet50V2(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        #================= Output - classification head ============================
        classification_output = Dense(self.num_classes, name="classification_head_before_activation")(x)
        classification_output = Activation('softmax', name="classification_head")(classification_output)

        #================= The final model ============================
        for layer in base_model.layers:
            layer.trainable = True
        model = Model(inputs=base_model.input, outputs=classification_output)
        return model
     
    def train(self, model):
        #================= Settings =========================
        def lr_scheduler(epoch):
            initial_lrate = 0.001#0.01#0.001#0.0005
            drop = 0.5
            epochs_drop = 30
            lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
            return lrate
        reduce_lr = LearningRateScheduler(lr_scheduler)
        weights_file = "{}{}{}".format(checkpoints_dir, "gray/", self.filename)
        model_checkpoint = ModelCheckpoint(weights_file, monitor='val_accuracy', save_best_only=True, verbose=1)
        csv_logger = CSVLogger("{}{}{}_history.csv".format(checkpoints_dir, "gray/", self.filename[:-3]), append=True)
        callbacks=[reduce_lr, model_checkpoint, csv_logger]

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
        opt = optimizers.SGD(lr=0, decay=1e-6, momentum=0.9, nesterov=True)  
        model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

        historytemp = model.fit_generator(datagen.flow(self.x_train, y=self.y_train, batch_size=self.batch_size),
                                          epochs=self.epochs, callbacks=callbacks,
                                          validation_data=(self.x_test, self.y_test))
        
        # #================= Save model and history =========================
        # with open("{}{}_history.pkl".format(checkpoints_dir, self.filename[:-3]), 'wb') as handle:
        #     pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # model.save_weights(weights_file)

        return model
