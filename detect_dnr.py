from __future__ import division, absolute_import, print_function
import argparse
from common.util import *
from setup_paths import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sfad.sfad_detector import *    


def main(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar', 'svhn', or 'tiny'"
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]
    assert os.path.isfile('{}cnn_{}.h5'.format(checkpoints_dir, args.dataset)), \
        'model file not found... must first train model using train_model.py.'
    
    layers = layer_names[DATASETS.index(args.dataset)]

    print('Loading the data and model...')
    # Load the model
    if args.dataset == 'mnist':
        from baselineCNN.cnn.cnn_mnist import MNISTCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        from dnr.dnr_mnist import DNRMNIST as dnr_class
        
    elif args.dataset == 'cifar':
        from baselineCNN.cnn.cnn_cifar10 import CIFAR10CNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        from dnr.dnr_cifar import DNRCIFAR as dnr_class

    elif args.dataset == 'svhn':
        from baselineCNN.cnn.cnn_svhn import SVHNCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        from dnr.dnr_svhn import DNRSVHN as dnr_class

    elif args.dataset == 'tiny':
        from baselineCNN.cnn.cnn_tiny import TINYCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        from dnr.dnr_tiny import DNRTINY as dnr_class

    # Load the dataset
    X_train, Y_train, X_test, Y_test = model_class.x_train, model_class.y_train, model_class.x_test, model_class.y_test

    #correctly classified samples
    predict_cnn = model.predict(X_test)
    inds_correct = np.where(predict_cnn.argmax(axis=1) == Y_test.argmax(axis=1))[0]
    X_test = X_test[inds_correct]
    Y_test = Y_test[inds_correct]

    #train and load classifiers
    detector_path = '{}dnr_{}.model'.format(dnr_results_dir,args.dataset)
    if not os.path.isfile(detector_path):
        dnr_class_clf = dnr_class(layer_names=layers)
    
    #load classifiers
    clfs = []
    for i in range(1,4):
        clfs.append(pickle.load(open(detector_path[:-6] + '_' + str(i) +'.model', 'rb')))
    clfs.append(pickle.load(open(detector_path, 'rb')))
    print('Classifiers loaded...')

    #get test data for the 3 classifiers and for the combiner classifier
    #data for the classifiers
    layers_data = []
    clf_outputs = np.array([])
    for layerx in layers:
        current = np.array([])
        for layer in layerx:
            inter_model = Model(inputs=model.get_input_at(0), outputs=model.get_layer(layer).output)
            if current.size==0:
                current = inter_model.predict(X_test).reshape((X_test.shape[0], -1))
            else:
                current = np.concatenate((current, inter_model.predict(X_test).reshape((X_test.shape[0], -1))), axis=1)
        layers_data.append(current)
    #data for the combiner
    for i in range(len(layers_data)):
        clf = clfs[i]
        if clf_outputs.size==0:
            clf_outputs = clf.predict_proba(layers_data[i])
        else:
            clf_outputs = np.concatenate((clf_outputs, clf.predict_proba(layers_data[i])), axis=1)
    
    #combiner output
    clf = clfs[-1]
    x_test_pred = clf.predict(clf_outputs)
    x_test_score = np.max(clf.predict_proba(clf_outputs), axis=1)

    #compute threshold
    thr = np.percentile(x_test_score, 10)

    #confidense detection
    reject_inds_clean = [i for i,v in enumerate(x_test_score) if np.max(x_test_score[i])<=thr]

    ## Evaluate detector - on adversarial attack
    for attack in ATTACKS:
        results_all = []

        #Prepare data
        # Load adversarial samples
        X_test_adv = np.load('{}{}_{}.npy'.format(adv_data_dir, args.dataset, attack))
        X_test_adv = X_test_adv[inds_correct]

        loss, acc_suc = model.evaluate(X_test_adv, Y_test, verbose=0)
        X_test_adv_pred = model.predict(X_test_adv)
        inds_success = np.where(X_test_adv_pred.argmax(axis=1) != Y_test.argmax(axis=1))[0]
        inds_fail = np.where(X_test_adv_pred.argmax(axis=1) == Y_test.argmax(axis=1))[0]

        #get adv data for the 3 classifiers and for the combiner classifier
        #adv data for the classifiers
        layers_adv = []
        clf_outputs_adv = np.array([])
        for layerx in layers:
            current = np.array([])
            for layer in layerx:
                inter_model = Model(inputs=model.get_input_at(0), outputs=model.get_layer(layer).output)
                if current.size==0:
                    current = inter_model.predict(X_test_adv).reshape((X_test_adv.shape[0], -1))
                else:
                    current = np.concatenate((current, inter_model.predict(X_test_adv).reshape((X_test_adv.shape[0], -1))), axis=1)
            layers_adv.append(current)
        #adv data for the combiner
        for i in range(len(layers_adv)):
            clf = clfs[i]
            if clf_outputs_adv.size==0:
                clf_outputs_adv = clf.predict_proba(layers_adv[i])
            else:
                clf_outputs_adv = np.concatenate((clf_outputs_adv, clf.predict_proba(layers_adv[i])), axis=1)
         
        #combiner output
        clf = clfs[-1]
        # x_adv_pred = clf.predict(clf_outputs_adv)
        x_adv_score = np.max(clf.predict_proba(clf_outputs_adv), axis=1)
        # x_adv_dec = np.max(clf.decision_function(clf_outputs_adv), axis=1)

        #confidense detection
        reject_inds_adv = [i for i,v in enumerate(x_adv_score) if np.max(x_adv_score[i])<=thr]

        #evaluation
        #For Y_all
        y_clean_pred = np.zeros(len(inds_correct), dtype=bool)
        y_clean_pred[reject_inds_clean] = True
        y_adv_pred = np.zeros(len(inds_correct), dtype=bool)
        y_adv_pred[reject_inds_adv] = True
        Y_all=np.concatenate((np.zeros(len(inds_correct), dtype=int), np.ones(len(inds_correct), dtype=int)))
        Y_all_pred=np.concatenate((y_clean_pred, y_adv_pred))
        # Y_all_pred_score=np.concatenate((x_test_score, x_adv_score))

        dec = np.ones(len(Y_all_pred))
        for i in range(len(dec)):
            if Y_all_pred[i]==False:
                dec[i]=-1

        acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all= evalulate_detection_test(Y_all, Y_all_pred)
        fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, dec)
        roc_auc_all = auc(fprs_all, tprs_all)
        print("AUC: {:.4f}%, Overall accuracy: {:.4f}%, FPR value: {:.4f}%".format(100*roc_auc_all, 100*acc_all, 100*fpr_all))

        curr_result = {'type':'all', 'nsamples': len(inds_correct),	'acc_suc': acc_suc,	\
                        'acc': acc_all, 'tpr': tpr_all, 'fpr': fpr_all, 'tp': tp_all, 'ap': ap_all, 'fb': fb_all, 'an': an_all,	\
                        'tprs': list(fprs_all), 'fprs': list(tprs_all),	'auc': roc_auc_all}
        results_all.append(curr_result)

        #For sucsess
        if len(inds_success)==0:
            tpr_success=np.nan
            curr_result = {'type':'success', 'nsamples': 0,	'acc_suc': 0,	\
                    'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                    'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
            results_all.append(curr_result)
        else:
            y_clean_success_pred=y_clean_pred[inds_success]
            y_adv_success_pred=y_adv_pred[inds_success]
            Y_success=np.concatenate((np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)))
            Y_success_pred=np.concatenate((y_clean_success_pred, y_adv_success_pred))
            # Y_success_pred_score=np.concatenate((x_test_score[inds_success], x_adv_score[inds_success]))

            dec = np.ones(len(Y_success_pred))
            for i in range(len(dec)):
                if Y_success_pred[i]==False:
                    dec[i]=-1

            accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(Y_success, Y_success_pred)
            fprs_success, tprs_success, thresholds_success = roc_curve(Y_success, dec)
            roc_auc_success = auc(fprs_success, tprs_success)

            curr_result = {'type':'success', 'nsamples': len(inds_success),	'acc_suc': 0,	\
                    'acc': accuracy_success, 'tpr': tpr_success, 'fpr': fpr_success, 'tp': tp_success, 'ap': ap_success, 'fb': fb_success, 'an': an_success,	\
                    'tprs': list(fprs_success), 'fprs': list(tprs_success),	'auc': roc_auc_success}
            results_all.append(curr_result)
        
        #For sucsess
        if len(inds_fail)==0:
            tpr_fail=np.nan
            curr_result = {'type':'fail', 'nsamples': 0,	'acc_suc': 0,	\
                    'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                    'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
            results_all.append(curr_result)
        else:
            y_clean_fail_pred=y_clean_pred[inds_fail]
            y_adv_fail_pred=y_adv_pred[inds_fail]
            Y_fail=np.concatenate((np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)))
            Y_fail_pred=np.concatenate((y_clean_fail_pred, y_adv_fail_pred))
            # Y_fail_pred_score=np.concatenate((x_test_score[inds_fail], x_adv_score[inds_fail]))

            dec = np.ones(len(Y_fail_pred))
            for i in range(len(dec)):
                if Y_fail_pred[i]==False:
                    dec[i]=-1

            accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
            fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, dec)
            roc_auc_fail = auc(fprs_fail, tprs_fail)

            curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                    'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                    'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
            results_all.append(curr_result)

        import csv
        with open('{}{}_{}.csv'.format(dnr_results_dir, args.dataset, attack), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_all:
                writer.writerow(row)
        
        print('{:>15} attack - accuracy of pretrained model: {:7.2f}% \
            - detection rates ------ SAEs: {:7.2f}%, FAEs: {:7.2f}%'.format(attack, 100*acc_suc, 100*tpr_success, 100*tpr_fail))
    
    print('Done!')

    # Gray-Box
    ## Evaluate detector - on adversarial attack
    for attack in ATTACKS:
        if not(attack=='hop' or attack=='sa' or attack=='sta' or (attack=='df' and args.dataset=='tiny')):
            results_all = []

            #Prepare data
            # Load adversarial samples
            X_test_adv = np.load('{}{}_{}.npy'.format(adv_data_gray_dir, args.dataset, attack))
            X_test_adv = X_test_adv[inds_correct]

            loss, acc_suc = model.evaluate(X_test_adv, Y_test, verbose=0)
            X_test_adv_pred = model.predict(X_test_adv)
            inds_success = np.where(X_test_adv_pred.argmax(axis=1) != Y_test.argmax(axis=1))[0]
            inds_fail = np.where(X_test_adv_pred.argmax(axis=1) == Y_test.argmax(axis=1))[0]

            #get adv data for the 3 classifiers and for the combiner classifier
            #adv data for the classifiers
            layers_adv = []
            clf_outputs_adv = np.array([])
            for layerx in layers:
                current = np.array([])
                for layer in layerx:
                    inter_model = Model(inputs=model.get_input_at(0), outputs=model.get_layer(layer).output)
                    if current.size==0:
                        current = inter_model.predict(X_test_adv).reshape((X_test_adv.shape[0], -1))
                    else:
                        current = np.concatenate((current, inter_model.predict(X_test_adv).reshape((X_test_adv.shape[0], -1))), axis=1)
                layers_adv.append(current)
            #adv data for the combiner
            for i in range(len(layers_adv)):
                clf = clfs[i]
                if clf_outputs_adv.size==0:
                    clf_outputs_adv = clf.predict_proba(layers_adv[i])
                else:
                    clf_outputs_adv = np.concatenate((clf_outputs_adv, clf.predict_proba(layers_adv[i])), axis=1)
            
            #combiner output
            clf = clfs[-1]
            # x_adv_pred = clf.predict(clf_outputs_adv)
            x_adv_score = np.max(clf.predict_proba(clf_outputs_adv), axis=1)
            # x_adv_dec = np.max(clf.decision_function(clf_outputs_adv), axis=1)

            #confidense detection
            reject_inds_adv = [i for i,v in enumerate(x_adv_score) if np.max(x_adv_score[i])<=thr]

            #evaluation
            #For Y_all
            y_clean_pred = np.zeros(len(inds_correct), dtype=bool)
            y_clean_pred[reject_inds_clean] = True
            y_adv_pred = np.zeros(len(inds_correct), dtype=bool)
            y_adv_pred[reject_inds_adv] = True
            Y_all=np.concatenate((np.zeros(len(inds_correct), dtype=int), np.ones(len(inds_correct), dtype=int)))
            Y_all_pred=np.concatenate((y_clean_pred, y_adv_pred))
            # Y_all_pred_score=np.concatenate((x_test_score, x_adv_score))

            dec = np.ones(len(Y_all_pred))
            for i in range(len(dec)):
                if Y_all_pred[i]==False:
                    dec[i]=-1

            acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all= evalulate_detection_test(Y_all, Y_all_pred)
            fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, dec)
            roc_auc_all = auc(fprs_all, tprs_all)
            print("AUC: {:.4f}%, Overall accuracy: {:.4f}%, FPR value: {:.4f}%".format(100*roc_auc_all, 100*acc_all, 100*fpr_all))

            curr_result = {'type':'all', 'nsamples': len(inds_correct),	'acc_suc': acc_suc,	\
                            'acc': acc_all, 'tpr': tpr_all, 'fpr': fpr_all, 'tp': tp_all, 'ap': ap_all, 'fb': fb_all, 'an': an_all,	\
                            'tprs': list(fprs_all), 'fprs': list(tprs_all),	'auc': roc_auc_all}
            results_all.append(curr_result)

            #For sucsess
            if len(inds_success)==0:
                tpr_success=np.nan
                curr_result = {'type':'success', 'nsamples': 0,	'acc_suc': 0,	\
                        'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                        'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
                results_all.append(curr_result)
            else:
                y_clean_success_pred=y_clean_pred[inds_success]
                y_adv_success_pred=y_adv_pred[inds_success]
                Y_success=np.concatenate((np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)))
                Y_success_pred=np.concatenate((y_clean_success_pred, y_adv_success_pred))
                # Y_success_pred_score=np.concatenate((x_test_score[inds_success], x_adv_score[inds_success]))

                dec = np.ones(len(Y_success_pred))
                for i in range(len(dec)):
                    if Y_success_pred[i]==False:
                        dec[i]=-1

                accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(Y_success, Y_success_pred)
                fprs_success, tprs_success, thresholds_success = roc_curve(Y_success, dec)
                roc_auc_success = auc(fprs_success, tprs_success)

                curr_result = {'type':'success', 'nsamples': len(inds_success),	'acc_suc': 0,	\
                        'acc': accuracy_success, 'tpr': tpr_success, 'fpr': fpr_success, 'tp': tp_success, 'ap': ap_success, 'fb': fb_success, 'an': an_success,	\
                        'tprs': list(fprs_success), 'fprs': list(tprs_success),	'auc': roc_auc_success}
                results_all.append(curr_result)
            
            #For sucsess
            if len(inds_fail)==0:
                tpr_fail=np.nan
                curr_result = {'type':'fail', 'nsamples': 0,	'acc_suc': 0,	\
                        'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                        'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
                results_all.append(curr_result)
            else:
                y_clean_fail_pred=y_clean_pred[inds_fail]
                y_adv_fail_pred=y_adv_pred[inds_fail]
                Y_fail=np.concatenate((np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)))
                Y_fail_pred=np.concatenate((y_clean_fail_pred, y_adv_fail_pred))
                # Y_fail_pred_score=np.concatenate((x_test_score[inds_fail], x_adv_score[inds_fail]))

                dec = np.ones(len(Y_fail_pred))
                for i in range(len(dec)):
                    if Y_fail_pred[i]==False:
                        dec[i]=-1

                accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
                fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, dec)
                roc_auc_fail = auc(fprs_fail, tprs_fail)

                curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                        'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                        'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
                results_all.append(curr_result)

            import csv
            with open('{}{}_gray_{}.csv'.format(dnr_results_gray_dir, args.dataset, attack), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results_all:
                    writer.writerow(row)
            
            print('Gray-box attack {}- accuracy of pretrained model: {:7.2f}% \
                - detection rates ------ SAEs: {:7.2f}%, FAEs: {:7.2f}%'.format(attack, 100*acc_suc, 100*tpr_success, 100*tpr_fail))
        
        print('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either {}".format(DATASETS),
        required=True, type=str
    )

    args = parser.parse_args()
    main(args)