from __future__ import division, absolute_import, print_function
import argparse
from common.util import *
from setup_paths import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.metrics import accuracy_score, precision_score, recall_score
from magnet.defensive_models import DenoisingAutoEncoder as DAE
from magnet.worker import *
from magnet.worker import *

def test(dic, X, thrs):
    dist_all = []
    pred_labels = []
    for d in dic:
        m = dic[d].mark(X)#m = np.reshape(dic[d].mark(X), (len(X),1))
        dist_all.append(m)
        pred_labels.append(m>thrs[d])
    
    #idx_pass = np.argwhere(marks < thrs[name])
    labels = pred_labels[0]
    for i in range(1, len(pred_labels)):
        labels = labels | pred_labels[i]
    
    # dist = dist_all[0]
    # for i in range(1, len(dist_all)):
    #     dist = np.max(np.concatenate((dist, dist_all[i]), axis=1), axis=1)
    
    return labels, dist_all 


def main(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar', 'svhn', or 'tiny'"
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]

    assert os.path.isfile('{}cnn_{}.h5'.format(checkpoints_dir, args.dataset)), \
        'model file not found... must first train model using train_model.py.'

    print('Loading the data and model...')
    # Load the model
    if args.dataset == 'mnist':
        from baselineCNN.cnn.cnn_mnist import MNISTCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        modelx = Model(inputs=model.get_input_at(0), outputs=model.get_layer('classification_head_before_activation').output)
        clip_min, clip_max = 0,1
        v_noise=0.1
        p1=2
        p2=1
        type='error'
        t=10
        drop_rate={"I": 0.001, "II": 0.001}
        epochs=100
        
    elif args.dataset == 'cifar':
        from baselineCNN.cnn.cnn_cifar10 import CIFAR10CNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        modelx = Model(inputs=model.get_input_at(0), outputs=model.get_layer('classification_head_before_softmax').output)
        clip_min, clip_max = 0,1
        v_noise=0.025
        p1=1
        p2=1
        type='error'
        t=40
        drop_rate={"I": 0.005, "II": 0.005}
        epochs=350

    elif args.dataset == 'svhn':
        from baselineCNN.cnn.cnn_svhn import SVHNCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        modelx = Model(inputs=model.get_input_at(0), outputs=model.get_layer('classification_head_before_activation').output)
        clip_min, clip_max = 0,1
        v_noise=0.025
        p1=1
        p2=1
        type='prob'
        t=40
        drop_rate={"I": 0.005, "II": 0.005}
        epochs=350

    elif args.dataset == 'tiny':
        from baselineCNN.cnn.cnn_tiny import TINYCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        modelx = Model(inputs=model.get_input_at(0), outputs=model.get_layer('classification_head_before_activation').output)
        # clip_min, clip_max = -2.117904,2.64
        clip_min, clip_max = 0,1
        v_noise=0.025
        p1=1
        p2=1
        type='error'
        t=10
        drop_rate={"I": 0.005, "II": 0.005}
        epochs=350

    # Load the dataset
    X_train, Y_train, X_test, Y_test = model_class.x_train, model_class.y_train, model_class.x_test, model_class.y_test
    val_size = 5000
    x_val = X_train[:val_size, :, :, :]
    y_val = Y_train[:val_size]
    X_train = X_train[val_size:, :, :, :]
    Y_train = Y_train[val_size:]

    #Train detector -- if already trained, load it
    detector_i_filename = '{}_magnet_detector_i.h5'.format(args.dataset)
    detector_ii_filename = '{}_magnet_detector_ii.h5'.format(args.dataset)
    im_dim = [X_train.shape[1], X_train.shape[2], X_train.shape[3]]
    detector_I = DAE(im_dim, [3, "average", 3], v_noise=v_noise, activation="sigmoid", model_dir=magnet_results_dir, reg_strength=1e-9)
    detector_II = DAE(im_dim, [3], v_noise=v_noise, activation="sigmoid", model_dir=magnet_results_dir, reg_strength=1e-9)
    if os.path.isfile('{}{}'.format(magnet_results_dir, detector_i_filename)):
        detector_I.load(detector_i_filename)
    else:
        detector_I.train(X_train, X_test, detector_i_filename, clip_min, clip_max, num_epochs=epochs, batch_size=256, if_save=True)
    if os.path.isfile('{}{}'.format(magnet_results_dir, detector_ii_filename)):
        detector_II.load(detector_ii_filename)
    else:
        detector_II.train(X_train, X_test, detector_ii_filename, clip_min, clip_max, num_epochs=epochs, batch_size=256, if_save=True)

    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    preds_test = model.predict(X_test)
    inds_correct = np.where(preds_test.argmax(axis=1) == Y_test.argmax(axis=1))[0]
    print("Number of correctly predict images: %s" % (len(inds_correct)))
    X_test = X_test[inds_correct]
    Y_test = Y_test[inds_correct]
    print("X_test: ", X_test.shape)

    #Make AEs ready
    classifier = Classifier(modelx, model_class.num_classes)
    if type=='error':
        if args.dataset=='cifar':
            detect_I = AEDetector(detector_I.model, p=p1)
            detect_II = AEDetector(detector_I.model, p=p2)
            reformer = SimpleReformer(detector_II.model)
        else:
            detect_I = AEDetector(detector_I.model, p=p1)
            detect_II = AEDetector(detector_II.model, p=p2)
            reformer = SimpleReformer(detector_I.model)
        detector_dict = dict()
        detector_dict["I"] = detect_I
        detector_dict["II"] = detect_II
    elif type=='prob':
        reformer = SimpleReformer(detector_I.model)
        reformer2 = SimpleReformer(detector_II.model)
        detect_I = DBDetector(reformer, reformer2, classifier, T=t)
        detector_dict = dict()
        detector_dict["I"] = detect_I

    operator = Operator(x_val, X_test, Y_test, classifier, detector_dict, reformer)

    ## Evaluate detector
    #on adversarial attack
    Y_test_copy=Y_test
    X_test_copy=X_test
    for attack in ATTACKS:
        Y_test=Y_test_copy
        X_test=X_test_copy
        results_all = []

        #Prepare data
        # Load adversarial samples
        X_test_adv = np.load('{}{}_{}.npy'.format(adv_data_dir, args.dataset, attack))
        if attack=='df' and args.dataset=='tiny':
            Y_test=model_class.y_test[0:2700]
            X_test=model_class.x_test[0:2700]
            cwi_inds = inds_correct[inds_correct<2700]
            Y_test = Y_test[cwi_inds]
            X_test = X_test[cwi_inds]
            X_test_adv = X_test_adv[cwi_inds]
        else:
            X_test_adv = X_test_adv[inds_correct]

        pred_adv = model.predict(X_test_adv)
        loss, acc_suc = model.evaluate(X_test_adv, Y_test, verbose=0)
        inds_success = np.where(pred_adv.argmax(axis=1) != Y_test.argmax(axis=1))[0]
        inds_fail = np.where(pred_adv.argmax(axis=1) == Y_test.argmax(axis=1))[0]
        X_test_adv_success = X_test_adv[inds_success]
        Y_test_success = Y_test[inds_success]
        X_test_adv_fail = X_test_adv[inds_fail]
        Y_test_fail = Y_test[inds_fail]

        # prepare X and Y for detectors
        X_all = np.concatenate([X_test, X_test_adv])
        Y_all = np.concatenate([np.zeros(len(X_test), dtype=bool), np.ones(len(X_test), dtype=bool)])
        X_success = np.concatenate([X_test[inds_success], X_test_adv_success])
        Y_success = np.concatenate([np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)])
        X_fail = np.concatenate([X_test[inds_fail], X_test_adv_fail])
        Y_fail = np.concatenate([np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

        # --- get thresholds per detector
        testAttack = AttackData(X_test_adv, np.argmax(Y_test, axis=1), attack)
        evaluator = Evaluator(operator, testAttack)
        thrs = evaluator.operator.get_thrs(drop_rate)

        #For Y_all 
        Y_all_pred, Y_all_pred_score = test(detector_dict, X_all, thrs)
        acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
        fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score[0])
        roc_auc_all = auc(fprs_all, tprs_all)
        print("AUC: {:.4f}%, Overall accuracy: {:.4f}%, FPR value: {:.4f}%".format(100*roc_auc_all, 100*acc_all, 100*fpr_all))

        curr_result = {'type':'all', 'nsamples': len(inds_correct),	'acc_suc': acc_suc,	\
                        'acc': acc_all, 'tpr': tpr_all, 'fpr': fpr_all, 'tp': tp_all, 'ap': ap_all, 'fb': fb_all, 'an': an_all,	\
                        'tprs': list(fprs_all), 'fprs': list(tprs_all),	'auc': roc_auc_all}
        results_all.append(curr_result)

        #for Y_success
        if len(inds_success)==0:
            tpr_success=np.nan
            curr_result = {'type':'success', 'nsamples': 0,	'acc_suc': 0,	\
                    'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                    'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
            results_all.append(curr_result)
        else:
            Y_success_pred, Y_success_pred_score = test(detector_dict, X_success, thrs)
            accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(Y_success, Y_success_pred)
            fprs_success, tprs_success, thresholds_success = roc_curve(Y_success, Y_success_pred_score[0])
            roc_auc_success = auc(fprs_success, tprs_success)

            curr_result = {'type':'success', 'nsamples': len(inds_success),	'acc_suc': 0,	\
                    'acc': accuracy_success, 'tpr': tpr_success, 'fpr': fpr_success, 'tp': tp_success, 'ap': ap_success, 'fb': fb_success, 'an': an_success,	\
                    'tprs': list(fprs_success), 'fprs': list(tprs_success),	'auc': roc_auc_success}
            results_all.append(curr_result)

        #for Y_fail
        if len(inds_fail)==0:
            tpr_fail=np.nan
            curr_result = {'type':'fail', 'nsamples': 0,	'acc_suc': 0,	\
                    'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                    'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
            results_all.append(curr_result)
        else:
            Y_fail_pred, Y_fail_pred_score = test(detector_dict, X_fail, thrs)
            accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
            fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, Y_fail_pred_score[0])
            roc_auc_fail = auc(fprs_fail, tprs_fail)

            curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                    'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                    'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
            results_all.append(curr_result)

        import csv
        with open('{}{}_{}.csv'.format(magnet_results_dir, args.dataset, attack), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_all:
                writer.writerow(row)
        
        print('{:>15} attack - accuracy of pretrained model: {:7.2f}% \
            - detection rates ------ SAEs: {:7.2f}%, FAEs: {:7.2f}%'.format(attack, 100*acc_suc, 100*tpr_success, 100*tpr_fail))
    
    print('Done!')

    #Gray box attack 
    ## Evaluate detector
    #on adversarial attack
    for attack in ATTACKS:
        if not(attack=='hop' or attack=='sa' or attack=='sta' or (attack=='df' and args.dataset=='tiny')):
            Y_test=Y_test_copy
            X_test=X_test_copy
            results_all = []

            #Prepare data
            # Load adversarial samples
            X_test_adv = np.load('{}{}_{}.npy'.format(adv_data_gray_dir, args.dataset, attack))
            if attack=='df' and args.dataset=='tiny':
                Y_test=model_class.y_test[0:2700]
                X_test=model_class.x_test[0:2700]
                cwi_inds = inds_correct[inds_correct<2700]
                Y_test = Y_test[cwi_inds]
                X_test = X_test[cwi_inds]
                X_test_adv = X_test_adv[cwi_inds]
            else:
                X_test_adv = X_test_adv[inds_correct]

            pred_adv = model.predict(X_test_adv)
            loss, acc_suc = model.evaluate(X_test_adv, Y_test, verbose=0)
            inds_success = np.where(pred_adv.argmax(axis=1) != Y_test.argmax(axis=1))[0]
            inds_fail = np.where(pred_adv.argmax(axis=1) == Y_test.argmax(axis=1))[0]
            X_test_adv_success = X_test_adv[inds_success]
            Y_test_success = Y_test[inds_success]
            X_test_adv_fail = X_test_adv[inds_fail]
            Y_test_fail = Y_test[inds_fail]

            # prepare X and Y for detectors
            X_all = np.concatenate([X_test, X_test_adv])
            Y_all = np.concatenate([np.zeros(len(X_test), dtype=bool), np.ones(len(X_test), dtype=bool)])
            X_success = np.concatenate([X_test[inds_success], X_test_adv_success])
            Y_success = np.concatenate([np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)])
            X_fail = np.concatenate([X_test[inds_fail], X_test_adv_fail])
            Y_fail = np.concatenate([np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

            # --- get thresholds per detector
            testAttack = AttackData(X_test_adv, np.argmax(Y_test, axis=1), attack)
            evaluator = Evaluator(operator, testAttack)
            thrs = evaluator.operator.get_thrs(drop_rate)

            #For Y_all 
            Y_all_pred, Y_all_pred_score = test(detector_dict, X_all, thrs)
            acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
            fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score[0])
            roc_auc_all = auc(fprs_all, tprs_all)
            print("AUC: {:.4f}%, Overall accuracy: {:.4f}%, FPR value: {:.4f}%".format(100*roc_auc_all, 100*acc_all, 100*fpr_all))

            curr_result = {'type':'all', 'nsamples': len(inds_correct),	'acc_suc': acc_suc,	\
                            'acc': acc_all, 'tpr': tpr_all, 'fpr': fpr_all, 'tp': tp_all, 'ap': ap_all, 'fb': fb_all, 'an': an_all,	\
                            'tprs': list(fprs_all), 'fprs': list(tprs_all),	'auc': roc_auc_all}
            results_all.append(curr_result)

            #for Y_success
            if len(inds_success)==0:
                tpr_success=np.nan
                curr_result = {'type':'success', 'nsamples': 0,	'acc_suc': 0,	\
                        'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                        'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
                results_all.append(curr_result)
            else:
                Y_success_pred, Y_success_pred_score = test(detector_dict, X_success, thrs)
                accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(Y_success, Y_success_pred)
                fprs_success, tprs_success, thresholds_success = roc_curve(Y_success, Y_success_pred_score[0])
                roc_auc_success = auc(fprs_success, tprs_success)

                curr_result = {'type':'success', 'nsamples': len(inds_success),	'acc_suc': 0,	\
                        'acc': accuracy_success, 'tpr': tpr_success, 'fpr': fpr_success, 'tp': tp_success, 'ap': ap_success, 'fb': fb_success, 'an': an_success,	\
                        'tprs': list(fprs_success), 'fprs': list(tprs_success),	'auc': roc_auc_success}
                results_all.append(curr_result)

            #for Y_fail
            if len(inds_fail)==0:
                tpr_fail=np.nan
                curr_result = {'type':'fail', 'nsamples': 0,	'acc_suc': 0,	\
                        'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                        'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
                results_all.append(curr_result)
            else:
                Y_fail_pred, Y_fail_pred_score = test(detector_dict, X_fail, thrs)
                accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
                fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, Y_fail_pred_score[0])
                roc_auc_fail = auc(fprs_fail, tprs_fail)

                curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                        'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                        'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
                results_all.append(curr_result)

            import csv
            with open('{}{}_gray_{}.csv'.format(magnet_results_gray_dir, args.dataset, attack), 'w', newline='') as csvfile:
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
    # parser.add_argument(
    #     '-a', '--attack',
    #     help="Attack to use train the discriminator; either  {}".format(ATTACK),
    #     required=False, type=str
    # )
    # parser.add_argument(
    #     '-b', '--batch_size',
    #     help="The batch size to use for training.",
    #     required=False, type=int
    # )

    # parser.set_defaults(batch_size=100)
    # parser.set_defaults(attack="cwi")
    args = parser.parse_args()
    main(args)