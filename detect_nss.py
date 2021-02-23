from __future__ import division, absolute_import, print_function
import argparse

from tensorflow.python.ops.gen_math_ops import floor
from common.util import *
from setup_paths import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from nss.MSCN import *
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale, MinMaxScaler

def main(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar', 'svhn', or 'tiny'"
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]
    assert os.path.isfile('{}cnn_{}.h5'.format(checkpoints_dir, args.dataset)), \
        'model file not found... must first train model using train_model.py.'

    pgd_per = pgd_percent[DATASETS.index(args.dataset)]

    print('Loading the data and model...')
    # Load the model
    if args.dataset == 'mnist':
        from baselineCNN.cnn.cnn_mnist import MNISTCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        
    elif args.dataset == 'cifar':
        from baselineCNN.cnn.cnn_cifar10 import CIFAR10CNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    elif args.dataset == 'svhn':
        from baselineCNN.cnn.cnn_svhn import SVHNCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    elif args.dataset == 'tiny':
        from baselineCNN.cnn.cnn_tiny import TINYCNN as myModel
        model_class = myModel(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        model=model_class.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    # Load the dataset
    X_train, Y_train, X_test, Y_test = model_class.x_train, model_class.y_train, model_class.x_test, model_class.y_test

    #-----------------------------------------------#
    #              Train NSS detector               #
    #-----------------------------------------------# 
    #extract nss features, from normal images
    x_train_f_path = '{}{}_normal_f.npy'.format(nss_results_dir, args.dataset)
    if not os.path.isfile(x_train_f_path):
        X_train_f = np.array([])
        for img in X_test:
            # parameters = calculate_ggd_aggd(img,'GGD', kernel_size=7, sigma=7/6)
            parameters = calculate_brisque_features(img)
            parameters = parameters.reshape((1,-1))
            if X_train_f.size==0:
                X_train_f = parameters
            else:
                X_train_f = np.concatenate((X_train_f, parameters), axis=0)
        np.save(x_train_f_path, X_train_f)
    else:
        X_train_f = np.load(x_train_f_path)
    

    # X_train_f = scale_features(X_train_f)
    # scaler = MinMaxScaler(feature_range=(-1,1)).fit(X_train_f)
    # X_train_f = scaler.transform(X_train_f)

    X_train_f_copy = X_train_f

    #extract nss features, from adversarial images -- PGD
    pgds = ['pgdi_0.03125', 'pgdi_0.0625', 'pgdi_0.125', 'pgdi_0.25', 'pgdi_0.3125', 'pgdi_0.5']
    adv_data_f_all = []
    for pgd in pgds:
        adv_data = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, pgd))
        adv_data_f_path = '{}{}_{}_f.npy'.format(nss_results_dir,args.dataset, pgd)
        if not os.path.isfile(adv_data_f_path):
            adv_data_f = np.array([])
            for img in adv_data:
                # parameters = calculate_ggd_aggd(img,'GGD', kernel_size=7, sigma=7/6)
                parameters = calculate_brisque_features(img)
                parameters = parameters.reshape((1,-1))
                if adv_data_f.size==0:
                    adv_data_f = parameters
                else:
                    adv_data_f = np.concatenate((adv_data_f, parameters), axis=0)
            np.save(adv_data_f_path, adv_data_f)
        else:
            adv_data_f = np.load(adv_data_f_path)
        
        # adv_data_f = scaler.transform(adv_data_f)
        adv_data_f_all.append(adv_data_f)

    #correctly classified samples
    preds_test = model.predict(X_test)
    inds_correct = np.where(preds_test.argmax(axis=1) == Y_test.argmax(axis=1))[0]
    X_test = X_test[inds_correct]
    Y_test = Y_test[inds_correct]
    X_train_f = X_train_f[inds_correct]
    for i in range(len(adv_data_f_all)):
        adv_data_f_all[i] = adv_data_f_all[i][inds_correct]
    
    # samples = [200, 200, 300, 100, 100, 100]
    samples = np.array(np.floor(np.array(pgd_per)*len(inds_correct)), dtype=np.int)

    inds_file = '{}{}_inds.npy'.format(nss_results_dir, args.dataset)
    if not os.path.isfile(inds_file):
        success_inds = []
        for pgd in pgds:
            adv_data = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, pgd))
            adv_data = adv_data[inds_correct]
            pred_adv = model.predict(adv_data)
            inds_success = np.where(pred_adv.argmax(axis=1) != Y_test.argmax(axis=1))[0]
            success_inds.append(inds_success)

        selected_inds = []
        inds = random.sample(list(success_inds[0]), samples[0])
        selected_inds.append(inds)
        for i in range(1, len(pgds)):
            all_inds=[]
            for j in range(len(selected_inds)):
                all_inds = np.concatenate((all_inds, selected_inds[j]))
            
            allowed_inds = list(set(success_inds[i])-set(all_inds))
            inds = random.sample(allowed_inds, np.min([samples[i], len(allowed_inds)]))
            selected_inds.append(inds)
        np.save(inds_file, selected_inds, allow_pickle=True)
    else:
        selected_inds = np.load(inds_file, allow_pickle=True)

    train_inds=[]
    for i in range(len(selected_inds)):
        train_inds = np.concatenate((train_inds, selected_inds[i]))
    train_inds = np.int32(train_inds)
    test_inds = np.asarray(list(set(range(len(inds_correct)))-set(train_inds)))
 
    #train the model
    x_normal_f = X_train_f[train_inds]
    y_normal_f = np.zeros(len(train_inds), dtype=np.uint8)
    x_adv_f = np.concatenate((adv_data_f_all[0][selected_inds[0]],\
                            adv_data_f_all[1][selected_inds[1]],\
                            adv_data_f_all[2][selected_inds[2]],\
                            adv_data_f_all[3][selected_inds[3]],\
                            adv_data_f_all[4][selected_inds[4]],\
                            adv_data_f_all[5][selected_inds[5]]))
    y_adv_f = np.ones(len(train_inds), dtype=np.uint8)
    
    x_train = np.concatenate((x_normal_f, x_adv_f))
    y_train = np.concatenate((y_normal_f, y_adv_f))

    min_ = np.min(x_train, axis=0)
    max_ = np.max(x_train, axis=0)
    x_train = scale_features(x_train, min_, max_)

    # scaler = MinMaxScaler(feature_range=(-1,1)).fit(x_train)
    # x_train = scaler.transform(x_train)

    # x_normal_f = X_train_f
    # y_normal_f = np.zeros(len(x_normal_f), dtype=np.uint8)
    # if args.dataset== 'mnist':
    #     x_adv_f = adv_data_f_all[3]
    # else:
    #     x_adv_f = adv_data_f_all[1]
    # y_adv_f = np.ones(len(x_adv_f), dtype=np.uint8)
    # x_train = np.concatenate((x_normal_f, x_adv_f))
    # y_train = np.concatenate((y_normal_f, y_adv_f))
    

    #mnist
    if args.dataset == 'mnist':
        c=1000000.0
        g=1e-08
    elif args.dataset == 'cifar':
        c=10000
        g=1e-05
    elif args.dataset == 'svhn':
        c=0.1
        g=1e-08
    else:
        c=10000000000
        g=0.0001
    clf = svm.SVC(C=10, kernel='sigmoid', gamma=0.01, probability=True,random_state=0)
    clf.fit(x_train, y_train)
    # pred_train = clf.predict(x_train)
    # prob_train = clf.predict_proba(x_train)
    # score_train = clf.score(x_train, y_train)
    # acc_train, _, fpr_train, _, _, _, _ = evalulate_detection_test(y_train, pred_train)

    # #Tuning
    # C_range = np.logspace(-2, 10, 13)
    # gamma_range = [1e-05]#np.logspace(-9, 3, 13)
    # param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=45)
    # grid = GridSearchCV(svm.SVC(kernel='sigmoid',random_state=45), param_grid=param_grid, cv=cv)
    # grid.fit(x_train, y_train)
    # #The best parameters are {'C': 100000.0, 'gamma': 1e-07} with a score of 1.00
    # print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    #-----------------------------------------------#
    #                 Evaluate NSS                  #
    #-----------------------------------------------# 
    ## Evaluate detector -- on adversarial attack
    Y_test_copy=Y_test
    X_test_copy=X_test
    X_train_f_copy=scale_features(X_train_f_copy, min_, max_)
    for attack in ATTACKS:
        Y_test=Y_test_copy
        X_test=X_test_copy
        X_train_f=X_train_f_copy
        results_all = []

        #Prepare data
        # Load adversarial samples
        X_test_adv = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, attack))
        #get NSS for adv
        adv_data_f_path = '{}{}_{}_f.npy'.format(nss_results_dir, args.dataset, attack)
        if not os.path.isfile(adv_data_f_path):
            adv_data_f = np.array([])
            for img in X_test_adv:
                # parameters = calculate_ggd_aggd(img,'GGD', kernel_size=7, sigma=7/6)
                parameters = calculate_brisque_features(img)
                parameters = parameters.reshape((1,-1))
                if adv_data_f.size==0:
                    adv_data_f = parameters
                else:
                    adv_data_f = np.concatenate((adv_data_f, parameters), axis=0)
            np.save(adv_data_f_path, adv_data_f)
        else:
            adv_data_f = np.load(adv_data_f_path)
        adv_data_f = scale_features(adv_data_f, min_, max_)
        # adv_data_f = scaler.transform(adv_data_f)
        
        if attack=='df' and args.dataset=='tiny':
            Y_test=model_class.y_test[0:2700]
            X_test=model_class.x_test[0:2700]
            X_train_f=X_train_f[0:2700]
            adv_data_f=adv_data_f[0:2700]
            X_test_adv = X_test_adv[0:2700]
            cwi_inds = inds_correct[inds_correct<2700]
            Y_test = Y_test[cwi_inds]
            X_test = X_test[cwi_inds]
            X_train_f=X_train_f[cwi_inds]
            X_test_adv = X_test_adv[cwi_inds]
            nss_adv = adv_data_f[cwi_inds]  
        else:
            X_test_adv = X_test_adv[inds_correct]
            nss_adv = adv_data_f[inds_correct]
            X_train_f = X_train_f[inds_correct]

        pred_adv = model.predict(X_test_adv)
        loss, acc_suc = model.evaluate(X_test_adv, Y_test, verbose=0)
        inds_success = np.where(pred_adv.argmax(axis=1) != Y_test.argmax(axis=1))[0]
        inds_fail = np.where(pred_adv.argmax(axis=1) == Y_test.argmax(axis=1))[0]     
        nss_adv_success = nss_adv[inds_success]
        nss_adv_fail = nss_adv[inds_fail]

        # prepare X and Y for detectors
        X_all = np.concatenate([X_train_f, nss_adv])
        Y_all = np.concatenate([np.zeros(len(X_train_f), dtype=bool), np.ones(len(X_train_f), dtype=bool)])
        X_success = np.concatenate([X_train_f[inds_success], nss_adv_success])
        Y_success = np.concatenate([np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)])
        X_fail = np.concatenate([X_train_f[inds_fail], nss_adv_fail])
        Y_fail = np.concatenate([np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

        #For Y_all
        if np.any(np.isnan(X_all)):
            X_all = np.nan_to_num(X_all)
        Y_all_pred = clf.predict(X_all)
        Y_all_pred_score = clf.decision_function(X_all)

        acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
        fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score)
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
            if np.any(np.isnan(X_success)):
                X_success = np.nan_to_num(X_success)
            Y_success_pred = clf.predict(X_success)
            Y_success_pred_score = clf.decision_function(X_success)
            accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(Y_success, Y_success_pred)
            fprs_success, tprs_success, thresholds_success = roc_curve(Y_success, Y_success_pred_score)
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
            if np.any(np.isnan(X_fail)):
                X_fail = np.nan_to_num(X_fail)
            Y_fail_pred = clf.predict(X_fail)
            Y_fail_pred_score = clf.decision_function(X_fail)
            accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
            fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, Y_fail_pred_score)
            roc_auc_fail = auc(fprs_fail, tprs_fail)

            curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                    'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                    'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
            results_all.append(curr_result)

        import csv
        with open('{}{}_{}.csv'.format(nss_results_dir, args.dataset, attack), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_all:
                writer.writerow(row)
        
        print('{:>15} attack - accuracy of pretrained model: {:7.2f}% \
            - detection rates ------ SAEs: {:7.2f}%, FAEs: {:7.2f}%'.format(attack, 100*acc_suc, 100*tpr_success, 100*tpr_fail))
    
    print('Done!')

    #Gray Box attacls
    ## Evaluate detector -- on adversarial attack
    for attack in ATTACKS:
        if not(attack=='hop' or attack=='sa' or attack=='sta' or (attack=='df' and args.dataset=='tiny')):
            Y_test=Y_test_copy
            X_test=X_test_copy
            X_train_f=X_train_f_copy
            results_all = []

            #Prepare data
            # Load adversarial samples
            X_test_adv = np.load('%s%s_%s.npy' % (adv_data_gray_dir, args.dataset, attack))
            #get NSS for adv
            adv_data_f_path = '{}{}_{}_f.npy'.format(nss_results_gray_dir, args.dataset, attack)
            if not os.path.isfile(adv_data_f_path):
                adv_data_f = np.array([])
                for img in X_test_adv:
                    # parameters = calculate_ggd_aggd(img,'GGD', kernel_size=7, sigma=7/6)
                    parameters = calculate_brisque_features(img)
                    parameters = parameters.reshape((1,-1))
                    if adv_data_f.size==0:
                        adv_data_f = parameters
                    else:
                        adv_data_f = np.concatenate((adv_data_f, parameters), axis=0)
                np.save(adv_data_f_path, adv_data_f)
            else:
                adv_data_f = np.load(adv_data_f_path)
            
            adv_data_f = scale_features(adv_data_f, min_, max_)
            # adv_data_f = scaler.transform(adv_data_f)
            
            if attack=='df' and args.dataset=='tiny':
                Y_test=model_class.y_test[0:2700]
                X_test=model_class.x_test[0:2700]
                X_train_f=X_train_f[0:2700]
                adv_data_f=adv_data_f[0:2700]
                X_test_adv = X_test_adv[0:2700]
                cwi_inds = inds_correct[inds_correct<2700]
                Y_test = Y_test[cwi_inds]
                X_test = X_test[cwi_inds]
                X_train_f=X_train_f[cwi_inds]
                X_test_adv = X_test_adv[cwi_inds]
                nss_adv = adv_data_f[cwi_inds]  
            else:
                X_test_adv = X_test_adv[inds_correct]
                nss_adv = adv_data_f[inds_correct]
                X_train_f = X_train_f[inds_correct]

            pred_adv = model.predict(X_test_adv)
            loss, acc_suc = model.evaluate(X_test_adv, Y_test, verbose=0)
            inds_success = np.where(pred_adv.argmax(axis=1) != Y_test.argmax(axis=1))[0]
            inds_fail = np.where(pred_adv.argmax(axis=1) == Y_test.argmax(axis=1))[0]     
            nss_adv_success = nss_adv[inds_success]
            nss_adv_fail = nss_adv[inds_fail]

            # prepare X and Y for detectors
            X_all = np.concatenate([X_train_f, nss_adv])
            Y_all = np.concatenate([np.zeros(len(X_train_f), dtype=bool), np.ones(len(X_train_f), dtype=bool)])
            X_success = np.concatenate([X_train_f[inds_success], nss_adv_success])
            Y_success = np.concatenate([np.zeros(len(inds_success), dtype=bool), np.ones(len(inds_success), dtype=bool)])
            X_fail = np.concatenate([X_train_f[inds_fail], nss_adv_fail])
            Y_fail = np.concatenate([np.zeros(len(inds_fail), dtype=bool), np.ones(len(inds_fail), dtype=bool)])

            #For Y_all
            if np.any(np.isnan(X_all)):
                X_all = np.nan_to_num(X_all)
            Y_all_pred = clf.predict(X_all)
            Y_all_pred_score = clf.decision_function(X_all)

            acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
            fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score)
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
                if np.any(np.isnan(X_success)):
                    X_success = np.nan_to_num(X_success)
                Y_success_pred = clf.predict(X_success)
                Y_success_pred_score = clf.decision_function(X_success)
                accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(Y_success, Y_success_pred)
                fprs_success, tprs_success, thresholds_success = roc_curve(Y_success, Y_success_pred_score)
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
                if np.any(np.isnan(X_fail)):
                    X_fail = np.nan_to_num(X_fail)
                Y_fail_pred = clf.predict(X_fail)
                Y_fail_pred_score = clf.decision_function(X_fail)
                accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
                fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, Y_fail_pred_score)
                roc_auc_fail = auc(fprs_fail, tprs_fail)

                curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                        'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                        'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
                results_all.append(curr_result)

            import csv
            with open('{}{}_gray_{}.csv'.format(nss_results_gray_dir, args.dataset, attack), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results_all:
                    writer.writerow(row)
            
            print('Gray {:>15} attack - accuracy of pretrained model: {:7.2f}% \
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
    #     help="Attack to use train the discriminator; either 'fgsm_eps', 'bim_eps', 'cw', 'pgd' 'deepfool'",
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