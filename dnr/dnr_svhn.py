from common.util import *
from setup_paths import *
from sklearn.svm import SVC
import keras

class DNRSVHN:
    def __init__(self, cnn_model= 'cnn_svhn.h5',  svm_clf_name="dnr_svhn.model", layer_names=[['l_17'], ['l_14'], ['l_10']]):
        self.cnn_model = cnn_model
        self.svm_clf_name = svm_clf_name
        self.layer_names = layer_names
        self.num_classes = 10

        #get data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_svhn_data()
        self.x_train, self.x_test = normalize_linear(self.x_train, self.x_test)

        #convert labels to one_hot
        self.y_test_labels = self.y_test
        self.y_train_labels = self.y_train
        self.y_train, self.y_test = toCat_onehot(self.y_train, self.y_test, self.num_classes)

        from baselineCNN.cnn.cnn_svhn import SVHNCNN as svhn_class
        svhn_model_class = svhn_class(mode="load", filename=self.cnn_model)
        self.svhn_model = svhn_model_class.model
        learning_rate = 0.1
        lr_decay = 1e-6
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        self.svhn_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        del svhn_model_class

        #data for the classifiers
        self.layers_data = []
        self.clf_outputs = np.array([])
        for layers in self.layer_names:
            current = np.array([])
            for layer in layers:
                inter_model = Model(inputs=self.svhn_model.get_input_at(0), outputs=self.svhn_model.get_layer(layer).output)
                if current.size==0:
                    current = inter_model.predict(self.x_train).reshape((self.x_train.shape[0], -1))
                else:
                    current = np.concatenate((current, inter_model.predict(self.x_train).reshape((self.x_train.shape[0], -1))), axis=1)
            self.layers_data.append(current)
        
        self.train()

    def train(self):
        
        for i in range(len(self.layers_data)):
            clf = SVC(probability=True, random_state=55)
            clf.fit(self.layers_data[i], self.y_train_labels)

            #save classifiers
            s = pickle.dumps(clf)
            f = open('{}{}'.format(dnr_results_dir, self.svm_clf_name[:-6] + '_' + str(i+1) +'.model'), "wb+")
            f.write(s)
            f.close()
            print('{} classifier training is finished'.format(str(i+1)))

            if self.clf_outputs.size==0:
                    self.clf_outputs = clf.predict_proba(self.layers_data[i])
            else:
                self.clf_outputs = np.concatenate((self.clf_outputs, clf.predict_proba(self.layers_data[i])), axis=1)
        
        clf = SVC(probability=True, random_state=55)
        clf.fit(self.clf_outputs, self.y_train_labels)

        #save classifiers
        s = pickle.dumps(clf)
        f = open('{}{}'.format(dnr_results_dir, self.svm_clf_name), "wb+")
        f.write(s)
        f.close()
        print('{} classifier training is finished'.format('Combiner'))
