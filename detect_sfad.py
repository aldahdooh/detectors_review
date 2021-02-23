from __future__ import division, absolute_import, print_function
import argparse
from common.util import *
from setup_paths import *
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sfad.sfad_detector import *    


def selective_loss(y_true, y_pred):
    loss = K.categorical_crossentropy(
        K.repeat_elements(
            y_pred[:, -1:], model_class_v7b.num_classes, axis=1) * y_true[:, :-1],
        y_pred[:, :-1]) + lamda * K.maximum(-K.mean(y_pred[:, -1]) + c, 0) ** 2
    return loss

def selective_acc(y_true, y_pred):
    g = K.cast(K.greater(y_pred[:, -1], 0.995), K.floatx())
    temp1 = K.sum(
        (g) * K.cast(K.equal(K.argmax(y_true[:, :-1], axis=-1), K.argmax(y_pred[:, :-1], axis=-1)), K.floatx()))
    temp1 = temp1 / K.sum(g)
    return K.cast(temp1, K.floatx())

def coverage(y_true, y_pred):
    g = K.cast(K.greater(y_pred[:, -1], 0.995), K.floatx())
    return K.mean(g)

def main(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either 'mnist', 'cifar', 'svhn', or 'tiny'"
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]
    assert os.path.isfile('{}cnn_{}.h5'.format(checkpoints_dir, args.dataset)), \
        'model file not found... must first train model using train_model.py.'

    no_defense_h5= 'cnn_{}.h5'.format(args.dataset)
    filename_a = 'sfad_{}_a.h5'.format(args.dataset)
    filename_b = 'sfad_{}_b.h5'.format(args.dataset)
    mode = 'train'

    if args.dataset == 'mnist':
        from sfad.sfad_mnist_b import sfad_b as sfad_classifier_b
        from sfad.sfad_mnist_a import sfad_a as sfad_classifier_a
        coverage = 1.0
        coverage_th_a = 0.995
        coverage_th_b = 0.7
        reject_ratio_c = 1.2
        reject_ratio_s = 6
    
    elif args.dataset == 'cifar':
        from sfad.sfad_cifar_b import sfad_b as sfad_classifier_b
        from sfad.sfad_cifar_a import sfad_a as sfad_classifier_a
        coverage = 0.9
        coverage_th_a = 0.9
        coverage_th_b = 0.7
        reject_ratio_c = 9
        reject_ratio_s = 4
    
    elif args.dataset == 'svhn':
        from sfad.sfad_svhn_b import sfad_b as sfad_classifier_b
        from sfad.sfad_svhn_a import sfad_a as sfad_classifier_a
        coverage = 0.9
        coverage_th_a = 0.9
        coverage_th_b = 0.7
        reject_ratio_c = 3
        reject_ratio_s = 4
    
    elif args.dataset == 'tiny':
        from sfad.sfad_tiny_b import sfad_b as sfad_classifier_b
        from sfad.sfad_tiny_a import sfad_a as sfad_classifier_a
        coverage = 0.8
        coverage_th_a = 0.5
        coverage_th_b = 0.5
        reject_ratio_c = 3
        reject_ratio_s = 2.5

    detector_b_path = '{}{}'.format(sfad_results_dir, filename_b)
    if not os.path.isfile(detector_b_path):
        sfad_a = sfad_classifier_a(mode=mode, no_defense_h5=no_defense_h5, filename=filename_a, coverage=coverage, coverage_th=coverage_th_a)
        sfad_b = sfad_classifier_b(mode=mode, no_defense_h5=no_defense_h5, filename_a=filename_a, filename_b=filename_b, coverage=coverage, coverage_th=coverage_th_b)
    else:
        mode='load'
        sfad_b = sfad_classifier_b(mode=mode, no_defense_h5=no_defense_h5, filename_a=filename_a, filename_b=filename_b, coverage=coverage, coverage_th=coverage_th_b)

    l1_name = layer_names[DATASETS.index(args.dataset)][0][0]
    l2_name = layer_names[DATASETS.index(args.dataset)][1][0]
    l3_name = layer_names[DATASETS.index(args.dataset)][2][0]
    #get correctly classified samples
    predict_cnn = sfad_b.model_class.no_defense_model.predict(sfad_b.model_class.x_test)
    inds_correct = np.where(predict_cnn.argmax(axis=1) == sfad_b.y_test_labels)[0]

    #clean samples and thresholds calculations
    model_a_1_clean = sfad_b.model_1.predict(sfad_b.model_class.l_1_test)[0][inds_correct, :]
    model_a_2_clean = sfad_b.model_2.predict(sfad_b.model_class.l_2_test)[0][inds_correct, :]
    model_a_3_clean = sfad_b.model_3.predict(sfad_b.model_class.l_3_test)[0][inds_correct, :]
    x_clean = np.concatenate((model_a_1_clean[:, :-1], model_a_2_clean[:, :-1], model_a_3_clean[:, :-1]), axis=1)
    model_b_clean = sfad_b.model.predict(x_clean)[0]

    #thresholds
    th_a_1_c = np.percentile(np.max(model_a_1_clean[:,:-1], axis=1), reject_ratio_c)
    th_a_2_c = np.percentile(np.max(model_a_2_clean[:,:-1], axis=1), reject_ratio_c)
    th_a_3_c = np.percentile(np.max(model_a_3_clean[:,:-1], axis=1), reject_ratio_c)
    th_b_c = np.percentile(np.max(model_b_clean[:,:-1], axis=1), reject_ratio_c)
    if args.dataset=='mnist':
        th_c = th_b_c
    else:
        th_c = np.max([th_a_1_c, th_a_2_c, th_a_3_c, th_b_c])

    th_a_1_s = np.percentile(model_a_1_clean[:,-1], reject_ratio_s)
    th_a_2_s = np.percentile(model_a_2_clean[:,-1], reject_ratio_s)
    th_a_3_s = np.percentile(model_a_3_clean[:,-1], reject_ratio_s)
    th_b_s = np.percentile(model_b_clean[:,-1], reject_ratio_s)

    #selection detection
    selecta1 = model_a_1_clean[:, -1] > th_a_1_s
    selecta2 = model_a_2_clean[:, -1] > th_a_2_s
    selecta3 = model_a_3_clean[:, -1] > th_a_3_s
    selectb = model_b_clean[:, -1] > th_b_s
    if args.dataset=='mnist':
        select = selectb
    else:
        select = np.logical_and(np.logical_and(selectb, selecta1), np.logical_and(selecta2, selecta3))
    select_inds_clean = [i for i,v in enumerate(select) if v==False]

    #confidense detection
    reject_inds_clean = [i for i,v in enumerate(model_b_clean) if np.max(model_b_clean[i,:-1])<=th_c]

    #ensemble detection
    ensemble_inds_clean = [i for i,v in enumerate(model_b_clean) if np.argmax(predict_cnn[inds_correct], axis=1)[i]!=np.argmax(model_b_clean[:,:-1], axis=1)[i]]

    #all detections
    reject_all_clean = np.array(np.unique(np.concatenate((select_inds_clean, reject_inds_clean, ensemble_inds_clean), axis=0)), dtype=int)
    
    ## Evaluate detector - on adversarial attack
    for attack in ATTACKS:
        results_all = []

        #Prepare data
        # Load adversarial samples
        X_test_adv = np.load('%s%s_%s.npy' % (adv_data_dir, args.dataset, attack))

        if attack=='df' and args.dataset=='tiny':
            y_test = sfad_b.y_test[0:2700]
            X_test_adv = X_test_adv[0:2700]
            cwi_inds = inds_correct[inds_correct<2700]
            y_test = y_test[cwi_inds,:-1]
            X_test_adv = X_test_adv[cwi_inds]
        else:
            X_test_adv = X_test_adv[inds_correct]
            y_test = sfad_b.y_test[inds_correct,:-1]

        loss, acc_suc = sfad_b.model_class.no_defense_model.evaluate(X_test_adv, y_test, verbose=0)
        X_test_adv_pred = sfad_b.model_class.no_defense_model.predict(X_test_adv)
        inds_success = np.where(X_test_adv_pred.argmax(axis=1) != y_test.argmax(axis=1))[0]
        inds_fail = np.where(X_test_adv_pred.argmax(axis=1) == y_test.argmax(axis=1))[0]

        intermediate_layer_model = Model(inputs=sfad_b.model_class.no_defense_model.input, outputs=sfad_b.model_class.no_defense_model.get_layer(l1_name).output)
        l_1 = intermediate_layer_model.predict(X_test_adv)
        if len(l_1.shape)==2:
            l_1 = l_1.reshape(l_1.shape[0],1, 1, l_1.shape[1])
        elif len(l_1.shape)==4:
            l_1 = l_1.reshape(l_1.shape[0],l_1.shape[1], l_1.shape[2], l_1.shape[3])
        intermediate_layer_model = Model(inputs=sfad_b.model_class.no_defense_model.input, outputs=sfad_b.model_class.no_defense_model.get_layer(l2_name).output)
        l_2 = intermediate_layer_model.predict(X_test_adv)
        if len(l_2.shape)==2:
            l_2 = l_2.reshape(l_2.shape[0],1, 1, l_2.shape[1])
        elif len(l_2.shape)==4:
            l_2 = l_2.reshape(l_2.shape[0],l_2.shape[1], l_2.shape[2], l_2.shape[3])
        intermediate_layer_model = Model(inputs=sfad_b.model_class.no_defense_model.input, outputs=sfad_b.model_class.no_defense_model.get_layer(l3_name).output)
        l_3 = intermediate_layer_model.predict(X_test_adv)
        if len(l_3.shape)==2:
            l_3 = l_3.reshape(l_3.shape[0],1, 1, l_3.shape[1])
        elif len(l_1.shape)==4:
            l_3 = l_3.reshape(l_3.shape[0],l_3.shape[1], l_3.shape[2], l_3.shape[3])
        model_a_1_adv = sfad_b.model_1.predict(l_1)[0]
        model_a_2_adv = sfad_b.model_2.predict(l_2)[0]
        model_a_3_adv = sfad_b.model_3.predict(l_3)[0]
        x_adv = np.concatenate((model_a_1_adv[:, :-1], model_a_2_adv[:, :-1], model_a_3_adv[:, :-1]), axis=1)

        model_b_adv = sfad_b.model.predict(x_adv)[0]
        #selection detection
        selecta1 = model_a_1_adv[:, -1] > th_a_1_s
        selecta2 = model_a_2_adv[:, -1] > th_a_2_s
        selecta3 = model_a_3_adv[:, -1] > th_a_3_s
        selectb = model_b_adv[:, -1] > th_b_s
        select = np.logical_and(np.logical_and(selectb, selecta1), np.logical_and(selecta2, selecta3))
        # select = np.logical_and(selecta1, np.logical_and(selecta2, selecta3))
        # select = selectb
        select_inds_adv = [i for i,v in enumerate(select) if v==False]

        #confidense detection
        reject_inds_adv = [i for i,v in enumerate(model_b_adv) if np.max(model_b_adv[i,:-1])<=th_c]

        #ensemble detection
        ensemble_inds_adv = [i for i,v in enumerate(model_b_adv) if np.argmax(X_test_adv_pred, axis=1)[i]!=np.argmax(model_b_adv[:,:-1], axis=1)[i]]

        #all detections
        reject_all_adv = np.array(np.unique(np.concatenate((select_inds_adv, reject_inds_adv, ensemble_inds_adv), axis=0)), dtype=int)

        #evaluation
        #For Y_all
        if attack=='df' and args.dataset=='tiny':
            y_clean_pred = np.zeros(len(x_adv), dtype=bool)
            #y_clean_pred[reject_all_clean] = True
        else:
            y_clean_pred = np.zeros(len(x_adv), dtype=bool)
            y_clean_pred[reject_all_clean] = True
        y_adv_pred = np.zeros(len(x_adv), dtype=bool)
        y_adv_pred[reject_all_adv] = True
        Y_all=np.concatenate((np.zeros(len(x_adv), dtype=int), np.ones(len(x_adv), dtype=int)))
        Y_all_pred=np.concatenate((y_clean_pred, y_adv_pred))
        # Y_all_pred_score=np.concatenate((model_b_clean[:, -1], model_b_adv[:, -1]))

        dec = np.ones(len(Y_all_pred))
        for i in range(len(dec)):
            if Y_all_pred[i]==0:
                dec[i]=-1

        acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
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
            # Y_success_pred_score=np.concatenate((model_b_clean[inds_success, -1], model_b_adv[inds_success, -1]))

            dec = np.ones(len(Y_success_pred))
            for i in range(len(dec)):
                if Y_success_pred[i]==0:
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
            # Y_fail_pred_score=np.concatenate((model_b_clean[inds_fail, -1], model_b_adv[inds_fail, -1]))

            dec = np.ones(len(Y_fail_pred))
            for i in range(len(dec)):
                if Y_fail_pred[i]==0:
                    dec[i]=-1

            accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
            fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, dec)
            roc_auc_fail = auc(fprs_fail, tprs_fail)

            curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                    'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                    'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
            results_all.append(curr_result)

        import csv
        with open('{}{}_{}.csv'.format(sfad_results_dir, args.dataset, attack), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_all:
                writer.writerow(row)
        
        print('{:>15} attack - accuracy of pretrained model: {:7.2f}% \
            - detection rates ------ SAEs: {:7.2f}%, FAEs: {:7.2f}%'.format(attack, 100*acc_suc, 100*tpr_success, 100*tpr_fail))
    
    print('Done!')

        ## Evaluate detector - on adversarial attack
    for attack in ATTACKS:
        if not(attack=='hop' or attack=='sa' or attack=='sta' or (attack=='df' and args.dataset=='tiny')):
            results_all = []

            #Prepare data
            # Load adversarial samples
            X_test_adv = np.load('%s%s_%s.npy' % (adv_data_gray_dir, args.dataset, attack))

            if attack=='df' and args.dataset=='tiny':
                y_test = sfad_b.y_test[0:2700]
                X_test_adv = X_test_adv[0:2700]
                cwi_inds = inds_correct[inds_correct<2700]
                y_test = y_test[cwi_inds,:-1]
                X_test_adv = X_test_adv[cwi_inds]
            else:
                X_test_adv = X_test_adv[inds_correct]
                y_test = sfad_b.y_test[inds_correct,:-1]

            loss, acc_suc = sfad_b.model_class.no_defense_model.evaluate(X_test_adv, y_test, verbose=0)
            X_test_adv_pred = sfad_b.model_class.no_defense_model.predict(X_test_adv)
            inds_success = np.where(X_test_adv_pred.argmax(axis=1) != y_test.argmax(axis=1))[0]
            inds_fail = np.where(X_test_adv_pred.argmax(axis=1) == y_test.argmax(axis=1))[0]

            intermediate_layer_model = Model(inputs=sfad_b.model_class.no_defense_model.input, outputs=sfad_b.model_class.no_defense_model.get_layer(l1_name).output)
            l_1 = intermediate_layer_model.predict(X_test_adv)
            if len(l_1.shape)==2:
                l_1 = l_1.reshape(l_1.shape[0],1, 1, l_1.shape[1])
            elif len(l_1.shape)==4:
                l_1 = l_1.reshape(l_1.shape[0],l_1.shape[1], l_1.shape[2], l_1.shape[3])
            intermediate_layer_model = Model(inputs=sfad_b.model_class.no_defense_model.input, outputs=sfad_b.model_class.no_defense_model.get_layer(l2_name).output)
            l_2 = intermediate_layer_model.predict(X_test_adv)
            if len(l_2.shape)==2:
                l_2 = l_2.reshape(l_2.shape[0],1, 1, l_2.shape[1])
            elif len(l_2.shape)==4:
                l_2 = l_2.reshape(l_2.shape[0],l_2.shape[1], l_2.shape[2], l_2.shape[3])
            intermediate_layer_model = Model(inputs=sfad_b.model_class.no_defense_model.input, outputs=sfad_b.model_class.no_defense_model.get_layer(l3_name).output)
            l_3 = intermediate_layer_model.predict(X_test_adv)
            if len(l_3.shape)==2:
                l_3 = l_3.reshape(l_3.shape[0],1, 1, l_3.shape[1])
            elif len(l_1.shape)==4:
                l_3 = l_3.reshape(l_3.shape[0],l_3.shape[1], l_3.shape[2], l_3.shape[3])
            model_a_1_adv = sfad_b.model_1.predict(l_1)[0]
            model_a_2_adv = sfad_b.model_2.predict(l_2)[0]
            model_a_3_adv = sfad_b.model_3.predict(l_3)[0]
            x_adv = np.concatenate((model_a_1_adv[:, :-1], model_a_2_adv[:, :-1], model_a_3_adv[:, :-1]), axis=1)

            model_b_adv = sfad_b.model.predict(x_adv)[0]
            #selection detection
            selecta1 = model_a_1_adv[:, -1] > th_a_1_s
            selecta2 = model_a_2_adv[:, -1] > th_a_2_s
            selecta3 = model_a_3_adv[:, -1] > th_a_3_s
            selectb = model_b_adv[:, -1] > th_b_s
            select = np.logical_and(np.logical_and(selectb, selecta1), np.logical_and(selecta2, selecta3))
            # select = np.logical_and(selecta1, np.logical_and(selecta2, selecta3))
            # select = selectb
            select_inds_adv = [i for i,v in enumerate(select) if v==False]

            #confidense detection
            reject_inds_adv = [i for i,v in enumerate(model_b_adv) if np.max(model_b_adv[i,:-1])<=th_c]

            #ensemble detection
            ensemble_inds_adv = [i for i,v in enumerate(model_b_adv) if np.argmax(X_test_adv_pred, axis=1)[i]!=np.argmax(model_b_adv[:,:-1], axis=1)[i]]

            #all detections
            reject_all_adv = np.array(np.unique(np.concatenate((select_inds_adv, reject_inds_adv, ensemble_inds_adv), axis=0)), dtype=int)

            #evaluation
            #For Y_all
            if attack=='df' and args.dataset=='tiny':
                y_clean_pred = np.zeros(len(x_adv), dtype=bool)
                #y_clean_pred[reject_all_clean] = True
            else:
                y_clean_pred = np.zeros(len(x_adv), dtype=bool)
                y_clean_pred[reject_all_clean] = True
            y_adv_pred = np.zeros(len(x_adv), dtype=bool)
            y_adv_pred[reject_all_adv] = True
            Y_all=np.concatenate((np.zeros(len(x_adv), dtype=int), np.ones(len(x_adv), dtype=int)))
            Y_all_pred=np.concatenate((y_clean_pred, y_adv_pred))
            # Y_all_pred_score=np.concatenate((model_b_clean[:, -1], model_b_adv[:, -1]))

            dec = np.ones(len(Y_all_pred))
            for i in range(len(dec)):
                if Y_all_pred[i]==0:
                    dec[i]=-1

            acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)
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
                # Y_success_pred_score=np.concatenate((model_b_clean[inds_success, -1], model_b_adv[inds_success, -1]))

                dec = np.ones(len(Y_success_pred))
                for i in range(len(dec)):
                    if Y_success_pred[i]==0:
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
                # Y_fail_pred_score=np.concatenate((model_b_clean[inds_fail, -1], model_b_adv[inds_fail, -1]))

                dec = np.ones(len(Y_fail_pred))
                for i in range(len(dec)):
                    if Y_fail_pred[i]==0:
                        dec[i]=-1

                accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(Y_fail, Y_fail_pred)
                fprs_fail, tprs_fail, thresholds_fail = roc_curve(Y_fail, dec)
                roc_auc_fail = auc(fprs_fail, tprs_fail)

                curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                        'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                        'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
                results_all.append(curr_result)

            import csv
            with open('{}{}_{}.csv'.format(sfad_results_gray_dir, args.dataset, attack), 'w', newline='') as csvfile:
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