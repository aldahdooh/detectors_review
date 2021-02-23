from __future__ import division, absolute_import, print_function
import argparse
from common.util import *
from setup_paths import *
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.neighbors import KernelDensity
from kd_bu.detect.util import get_data, get_noisy_samples, get_mc_predictions, get_deep_representations, score_samples, normalize, normalize_std, train_lr, compute_roc
from sklearn.preprocessing import scale, StandardScaler

def main(args):
    assert args.dataset in DATASETS, \
        "Dataset parameter must be either {}".format(DATASETS)
    ATTACKS = ATTACK[DATASETS.index(args.dataset)]
    assert args.attack in ATTACKS, \
        "Train attack must be either {}".format(ATTACKS)
    assert os.path.isfile('{}cnn_{}.h5'.format(checkpoints_dir, args.dataset)), \
        'model file not found... must first train model'
    assert os.path.isfile('{}{}_{}.npy'.format(adv_data_dir, args.dataset, args.attack)), \
        'adversarial sample file not found... must first craft adversarial samples'
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

    if args.attack=='df' and args.dataset=='tiny':
        X_test = X_test[0:2700] 
        Y_test = Y_test[0:2700]

    # Refine the normal, noisy and adversarial sets to only include samples for
    # which the original version was correctly classified by the model
    preds_test = model.predict(X_test)
    inds_correct = np.where(preds_test.argmax(axis=1) == Y_test.argmax(axis=1))[0]
    X_test = X_test[inds_correct]
    Y_test = Y_test[inds_correct]
    # n_samples = len(inds_correct)
    
    # Check attack type, select adversarial and noisy samples accordingly
    print('Loading noisy and adversarial samples...')
    # Load adversarial samples
    X_test_adv = np.load('{}{}_{}.npy'.format(adv_data_dir, args.dataset, args.attack))
    X_test_adv = X_test_adv[inds_correct]
    # Craft an equal number of noisy samples
    X_test_noisy = get_noisy_samples(X_test, X_test_adv, args.dataset, args.attack)
    
    ## Get Bayesian uncertainty scores
    print('Getting Monte Carlo dropout variance predictions...')
    if args.attack=='df' and args.dataset=='tiny':
        uncerts_normal_file = '{}{}_uncerts_normal_min.npy'.format(kd_bu_results_dir, args.dataset)
        if os.path.isfile(uncerts_normal_file):
            uncerts_normal = np.load(uncerts_normal_file)
        else:
            uncerts_normal = get_mc_predictions(model, X_test, batch_size=args.batch_size).var(axis=0).mean(axis=1)
            np.save(uncerts_normal_file, uncerts_normal)
    else:
        uncerts_normal_file = '{}{}_uncerts_normal.npy'.format(kd_bu_results_dir, args.dataset)
        if os.path.isfile(uncerts_normal_file):
            uncerts_normal = np.load(uncerts_normal_file)
        else:
            uncerts_normal = get_mc_predictions(model, X_test, batch_size=args.batch_size).var(axis=0).mean(axis=1)
            np.save(uncerts_normal_file, uncerts_normal)

    uncerts_noisy_file = '{}{}_{}_uncerts_noisy.npy'.format(kd_bu_results_dir, args.dataset, args.attack)
    if os.path.isfile(uncerts_noisy_file):
        uncerts_noisy = np.load(uncerts_noisy_file)
    else:
        uncerts_noisy = get_mc_predictions(model, X_test_noisy, batch_size=args.batch_size).var(axis=0).mean(axis=1)
        np.save(uncerts_noisy_file, uncerts_noisy)

    uncerts_adv_file = '{}{}_{}_uncerts_adv.npy'.format(kd_bu_results_dir, args.dataset, args.attack)
    if os.path.isfile(uncerts_adv_file):
        uncerts_adv = np.load(uncerts_adv_file)
    else:
        uncerts_adv = get_mc_predictions(model, X_test_adv, batch_size=args.batch_size).var(axis=0).mean(axis=1)
        np.save(uncerts_adv_file, uncerts_adv)

    ## Get KDE scores
    # Get deep feature representations
    print('Getting deep feature representations...')
    X_train_features_file = '{}{}_dens_xtrain.npy'.format(kd_bu_results_dir, args.dataset)
    if os.path.isfile(X_train_features_file):
        X_train_features = np.load(X_train_features_file)
    else:
        X_train_features = get_deep_representations(model, X_train, batch_size=args.batch_size, dataset=args.dataset)
        np.save(X_train_features_file, X_train_features)
    
    if args.attack=='df' and args.dataset=='tiny':
        X_test_normal_features_file = '{}{}_dens_normal_mini.npy'.format(kd_bu_results_dir, args.dataset, dataset=args.dataset)
        if os.path.isfile(X_test_normal_features_file):
            X_test_normal_features = np.load(X_test_normal_features_file)
        else:
            X_test_normal_features = get_deep_representations(model, X_test, batch_size=args.batch_size, dataset=args.dataset)
            np.save(X_test_normal_features_file, X_test_normal_features)
    else:
        X_test_normal_features_file = '{}{}_dens_normal.npy'.format(kd_bu_results_dir, args.dataset, dataset=args.dataset)
        if os.path.isfile(X_test_normal_features_file):
            X_test_normal_features = np.load(X_test_normal_features_file)
        else:
            X_test_normal_features = get_deep_representations(model, X_test, batch_size=args.batch_size, dataset=args.dataset)
            np.save(X_test_normal_features_file, X_test_normal_features)

    X_test_noisy_features_file = '{}{}_{}_dens_noisy.npy'.format(kd_bu_results_dir, args.dataset, args.attack)
    if os.path.isfile(X_test_noisy_features_file):
        X_test_noisy_features = np.load(X_test_noisy_features_file)
    else:
        X_test_noisy_features = get_deep_representations(model, X_test_noisy, batch_size=args.batch_size, dataset=args.dataset)
        np.save(X_test_noisy_features_file, X_test_noisy_features)

    X_test_adv_features_file = '{}{}_{}_dens_adv.npy'.format(kd_bu_results_dir, args.dataset, args.attack)
    if os.path.isfile(X_test_adv_features_file):
        X_test_adv_features = np.load(X_test_adv_features_file)
    else:
        X_test_adv_features = get_deep_representations(model, X_test_adv, batch_size=args.batch_size, dataset=args.dataset)
        np.save(X_test_adv_features_file, X_test_adv_features)

    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the "
                  "bandwidth.")
    for i in range(Y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=BANDWIDTHS[args.dataset]) \
            .fit(X_train_features[class_inds[i]])
    # Get model predictions
    print('Computing model predictions...')
    preds_test_normal = model.predict(X_test, verbose=0)
    preds_test_normal = preds_test_normal.argmax(axis=1)
    preds_test_noisy = model.predict(X_test_noisy)
    preds_test_noisy = preds_test_noisy.argmax(axis=1)
    preds_test_adv = model.predict(X_test_adv)
    preds_test_adv = preds_test_adv.argmax(axis=1)

    # Get density estimates
    print('computing densities...')
    densities_normal = score_samples(
        kdes,
        X_test_normal_features,
        preds_test_normal
    )
    densities_noisy = score_samples(
        kdes,
        X_test_noisy_features,
        preds_test_noisy
    )
    densities_adv = score_samples(
        kdes,
        X_test_adv_features,
        preds_test_adv
    )

    ## Z-score the uncertainty and density values
    uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z, uncerts_scaler = normalize_std(
        uncerts_normal,
        uncerts_adv,
        uncerts_noisy
    )
    densities_normal_z, densities_adv_z, densities_noisy_z, dense_scaler = normalize_std(
        densities_normal,
        densities_adv,
        densities_noisy
    )
    
    #70% train  --- 30% test
    indx_start = int(len(X_test_adv)*0.007)*100

    ## Build detector
    values, labels, lr = train_lr(
        densities_pos=densities_adv_z[:indx_start],
        densities_neg=np.concatenate((densities_normal_z[:indx_start], densities_noisy_z[:indx_start])),
        uncerts_pos=uncerts_adv_z[:indx_start],
        uncerts_neg=np.concatenate((uncerts_normal_z[:indx_start], uncerts_noisy_z[:indx_start]))
    )

    ## Evaluate detector on test samples
    preds_test_adv = model.predict(X_test_adv[indx_start:])
    preds_test_adv = preds_test_adv.argmax(axis=1)
    _, acc_suc = model.evaluate(X_test_adv[indx_start:], Y_test[indx_start:], verbose=0)
    inds_success = np.where(preds_test_adv != Y_test[indx_start:].argmax(axis=1))[0]
    inds_fail = np.where(preds_test_adv == Y_test[indx_start:].argmax(axis=1))[0]

    #For all
    values_pos = np.concatenate((densities_adv_z[indx_start:].reshape((1, -1)), uncerts_adv_z[indx_start:].reshape((1, -1))),  axis=0).transpose([1, 0])
    values_normal = np.concatenate((densities_normal_z[indx_start:].reshape((1, -1)), uncerts_normal_z[indx_start:].reshape((1, -1))),  axis=0).transpose([1, 0])
    # values_noise = np.concatenate((densities_noisy_z[indx_start:].reshape((1, -1)), uncerts_noisy_z[indx_start:].reshape((1, -1))),  axis=0).transpose([1, 0])
    # values_neg = np.concatenate((values_normal, values_noise))
    values_neg = values_normal
    values = np.concatenate((values_neg, values_pos))
    # labels = np.concatenate((np.zeros(len(values_normal)*2), np.ones(len(values_pos))))
    labels = np.concatenate((np.zeros(len(values_normal)), np.ones(len(values_pos))))
    
    results_all = []
    probs = lr.predict_proba(values)[:, 1]
    y_label_pred = lr.predict(values)

    acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(labels, y_label_pred)
    fprs_all, tprs_all, thresholds_all = roc_curve(labels, probs)
    roc_auc_all = auc(fprs_all, tprs_all)
    print("AUC: {:.4f}%, Overall accuracy: {:.4f}%, FPR value: {:.4f}%".format(100*roc_auc_all, 100*acc_all, 100*fpr_all))

    curr_result = {'type':'all', 'nsamples': len(inds_correct),	'acc_suc': acc_suc,	\
                    'acc': acc_all, 'tpr': tpr_all, 'fpr': fpr_all, 'tp': tp_all, 'ap': ap_all, 'fb': fb_all, 'an': an_all,	\
                    'tprs': list(fprs_all), 'fprs': list(tprs_all),	'auc': roc_auc_all}
    results_all.append(curr_result)

    #Predict Success
    if len(inds_success)==0:
        tpr_success=np.nan
        curr_result = {'type':'success', 'nsamples': 0,	'acc_suc': 0,	\
                    'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                    'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
        results_all.append(curr_result)
    else:
        values_success_pos = np.concatenate((densities_adv_z[indx_start:][inds_success].reshape((1, -1)), uncerts_adv_z[indx_start:][inds_success].reshape((1, -1))),  axis=0).transpose([1, 0])
        values_success_normal = np.concatenate((densities_normal_z[indx_start:][inds_success].reshape((1, -1)), uncerts_normal_z[indx_start:][inds_success].reshape((1, -1))),  axis=0).transpose([1, 0])
        # values_success_noise = np.concatenate((densities_noisy_z[indx_start:][inds_success].reshape((1, -1)), uncerts_noisy_z[indx_start:][inds_success].reshape((1, -1))),  axis=0).transpose([1, 0])
        # values_success_neg = np.concatenate((values_success_normal, values_success_noise))
        values_success_neg = values_success_normal
        values_success = np.concatenate((values_success_neg, values_success_pos))
        # labels_success = np.concatenate((np.zeros(len(inds_success)*2), np.ones(len(inds_success))))
        labels_success = np.concatenate((np.zeros(len(inds_success)), np.ones(len(inds_success))))

        probs_success = lr.predict_proba(values_success)[:, 1]
        y_label_pred_success = lr.predict(values_success)

        accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(labels_success, y_label_pred_success)
        fprs_success, tprs_success, thresholds_success = roc_curve(labels_success, probs_success)
        roc_auc_success = auc(fprs_success, tprs_success)
    
        curr_result = {'type':'success', 'nsamples': len(inds_success),	'acc_suc': 0,	\
                    'acc': accuracy_success, 'tpr': tpr_success, 'fpr': fpr_success, 'tp': tp_success, 'ap': ap_success, 'fb': fb_success, 'an': an_success,	\
                    'tprs': list(fprs_success), 'fprs': list(tprs_success),	'auc': roc_auc_success}
        results_all.append(curr_result)

    #Predict Fail
    if len(inds_fail)==0:
        tpr_fail=np.nan
        curr_result = {'type':'fail', 'nsamples': 0,	'acc_suc': 0,	\
                    'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                    'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
        results_all.append(curr_result)
    else:
        values_fail_pos = np.concatenate((densities_adv_z[indx_start:][inds_fail].reshape((1, -1)), uncerts_adv_z[indx_start:][inds_fail].reshape((1, -1))),  axis=0).transpose([1, 0])
        values_fail_normal = np.concatenate((densities_normal_z[indx_start:][inds_fail].reshape((1, -1)), uncerts_normal_z[indx_start:][inds_fail].reshape((1, -1))),  axis=0).transpose([1, 0])
        # values_fail_noise = np.concatenate((densities_noisy_z[indx_start:][inds_fail].reshape((1, -1)), uncerts_noisy_z[indx_start:][inds_fail].reshape((1, -1))),  axis=0).transpose([1, 0])
        # values_fail_neg = np.concatenate((values_fail_normal, values_fail_noise))
        values_fail_neg = values_fail_normal
        values_fail = np.concatenate((values_fail_neg, values_fail_pos))
        # labels_fail = np.concatenate((np.zeros(len(inds_fail)*2), np.ones(len(inds_fail))))
        labels_fail = np.concatenate((np.zeros(len(inds_fail)), np.ones(len(inds_fail))))

        probs_fail = lr.predict_proba(values_fail)[:, 1]
        y_label_pred_fail = lr.predict(values_fail)

        accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(labels_fail, y_label_pred_fail)
        fprs_fail, tprs_fail, thresholds_fail = roc_curve(labels_fail, probs_fail)
        roc_auc_fail = auc(fprs_fail, tprs_fail)

        curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                    'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                    'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
        results_all.append(curr_result)

    print('{:>15} attack - accuracy of pretrained model: {:7.2f}% \
        - detection rates ------ SAEs: {:7.2f}%, FAEs: {:7.2f}%'.format(args.attack, 100*acc_suc, 100*tpr_success, 100*tpr_fail))

    import csv
    with open('{}{}_{}.csv'.format(kd_bu_results_dir, args.dataset, args.attack), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_all:
            writer.writerow(row)
    
    print('Done!')

    # #use current detector to predict other attacks
    # Y_test_copy=Y_test
    # for attack in ATTACKS:
    #     Y_test=Y_test_copy
    #     if attack=='df'  and args.dataset=='tiny':
    #         Y_test=model_class.y_test[0:2700]
    #     results_all = []
    #     #Prepare data
    #     # Load adversarial samples
    #     adv_path = '{}{}_{}.npy'.format(adv_data_dir, args.dataset, attack)
    #     X_test_adv = np.load(adv_path)

    #     if attack=='df' and args.dataset=='tiny':
    #         cwi_inds = inds_correct[inds_correct<2700]
    #         Y_test = Y_test[cwi_inds]
    #         X_test_adv = X_test_adv[cwi_inds]
    #         X_test_noisy = get_noisy_samples(X_test[0:2700][cwi_inds], X_test_adv, args.dataset, attack)
    #     else:
    #         X_test_adv = X_test_adv[inds_correct]
    #         X_test_noisy = get_noisy_samples(X_test, X_test_adv, args.dataset, attack)

    #     uncerts_adv_file = '{}{}_{}_uncerts_adv.npy'.format(kd_bu_results_dir, args.dataset, attack)
    #     if os.path.isfile(uncerts_adv_file):
    #         uncerts_adv = np.load(uncerts_adv_file)
    #     else:
    #         uncerts_adv = get_mc_predictions(model, X_test_adv, batch_size=args.batch_size).var(axis=0).mean(axis=1)
    #         np.save(uncerts_adv_file, uncerts_adv)

    #     X_test_adv_features_file = '{}{}_{}_dens_adv.npy'.format(kd_bu_results_dir, args.dataset, attack)
    #     if os.path.isfile(X_test_adv_features_file):
    #         X_test_adv_features = np.load(X_test_adv_features_file)
    #     else:
    #         X_test_adv_features = get_deep_representations(model, X_test_adv, batch_size=args.batch_size, dataset=args.dataset)
    #         np.save(X_test_adv_features_file, X_test_adv_features)
        
    #     # Get model predictions
    #     preds_test_adv = model.predict(X_test_adv)
    #     loss, acc_suc = model.evaluate(X_test_adv, Y_test, verbose=0)
    #     preds_test_adv = preds_test_adv.argmax(axis=1)
    #     inds_success = np.where(preds_test_adv != Y_test.argmax(axis=1))[0]
    #     inds_fail = np.where(preds_test_adv == Y_test.argmax(axis=1))[0]

    #     # Get density estimates
    #     densities_adv = score_samples(kdes, X_test_adv_features, preds_test_adv)

    #     # Z-score the uncertainty and density values
    #     uncerts_adv_z = uncerts_scaler.transform(uncerts_adv.reshape((-1,1))).reshape((-1,))
    #     densities_adv_z = dense_scaler.transform(densities_adv.reshape((-1,1))).reshape((-1,))

    #     #Predict all
    #     values_pos = np.concatenate((densities_adv_z.reshape((1, -1)), uncerts_adv_z.reshape((1, -1))),  axis=0).transpose([1, 0])
    #     values_normal = np.concatenate((densities_normal_z.reshape((1, -1)), uncerts_normal_z.reshape((1, -1))),  axis=0).transpose([1, 0])
    #     values_noise = np.concatenate((densities_noisy_z.reshape((1, -1)), uncerts_noisy_z.reshape((1, -1))),  axis=0).transpose([1, 0])
    #     values_neg = values_normal
    #     values = np.concatenate((values_neg, values_pos))
    #     labels = np.concatenate((np.zeros(len(values_normal)), np.ones(len(values_pos))))
    #     probs = lr.predict_proba(values)[:, 1]
    #     y_label_pred = lr.predict(values)

    #     acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all= evalulate_detection_test(labels, y_label_pred)
    #     fprs_all, tprs_all, thresholds_all = roc_curve(labels, probs)
    #     roc_auc_all = auc(fprs_all, tprs_all)
    #     print("AUC: {:.4f}%, Overall accuracy: {:.4f}%, FPR value: {:.4f}%".format(100*roc_auc_all, 100*acc_all, 100*fpr_all))

    #     curr_result = {'type':'all', 'nsamples': len(inds_correct),	'acc_suc': acc_suc,	\
    #                     'acc': acc_all, 'tpr': tpr_all, 'fpr': fpr_all, 'tp': tp_all, 'ap': ap_all, 'fb': fb_all, 'an': an_all,	\
    #                     'tprs': list(fprs_all), 'fprs': list(tprs_all),	'auc': roc_auc_all}
    #     results_all.append(curr_result)

    #     #Predict Success
    #     if len(inds_success)==0:
    #         tpr_success=np.nan
    #         curr_result = {'type':'success', 'nsamples': 0,	'acc_suc': 0,	\
    #                 'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
    #                 'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
    #         results_all.append(curr_result)
    #     else:
    #         values_success_pos = np.concatenate((densities_adv_z[inds_success].reshape((1, -1)), uncerts_adv_z[inds_success].reshape((1, -1))),  axis=0).transpose([1, 0])
    #         values_success_normal = np.concatenate((densities_normal_z[inds_success].reshape((1, -1)), uncerts_normal_z[inds_success].reshape((1, -1))),  axis=0).transpose([1, 0])
    #         values_success_noise = np.concatenate((densities_noisy_z[inds_success].reshape((1, -1)), uncerts_noisy_z[inds_success].reshape((1, -1))),  axis=0).transpose([1, 0])
    #         # values_success_neg = np.concatenate((values_success_normal, values_success_noise))
    #         values_success_neg = values_success_normal
    #         values_success = np.concatenate((values_success_neg, values_success_pos))
    #         # labels_success = np.concatenate((np.zeros(len(inds_success)*2), np.ones(len(inds_success))))
    #         labels_success = np.concatenate((np.zeros(len(inds_success)), np.ones(len(inds_success))))
    #         probs_success = lr.predict_proba(values_success)[:, 1]
    #         y_label_pred_success = lr.predict(values_success)

    #         accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(labels_success, y_label_pred_success)
    #         fprs_success, tprs_success, thresholds_success = roc_curve(labels_success, probs_success)
    #         roc_auc_success = auc(fprs_success, tprs_success)

    #         curr_result = {'type':'success', 'nsamples': len(inds_success),	'acc_suc': 0,	\
    #                 'acc': accuracy_success, 'tpr': tpr_success, 'fpr': fpr_success, 'tp': tp_success, 'ap': ap_success, 'fb': fb_success, 'an': an_success,	\
    #                 'tprs': list(fprs_success), 'fprs': list(tprs_success),	'auc': roc_auc_success}
    #         results_all.append(curr_result)
        
    #     #Predict Fail
    #     if len(inds_fail)==0:
    #         tpr_fail=np.nan
    #         curr_result = {'type':'fail', 'nsamples': 0,	'acc_suc': 0,	\
    #                 'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
    #                 'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
    #         results_all.append(curr_result)
    #     else:
    #         values_fail_pos = np.concatenate((densities_adv_z[inds_fail].reshape((1, -1)), uncerts_adv_z[inds_fail].reshape((1, -1))),  axis=0).transpose([1, 0])
    #         values_fail_normal = np.concatenate((densities_normal_z[inds_fail].reshape((1, -1)), uncerts_normal_z[inds_fail].reshape((1, -1))),  axis=0).transpose([1, 0])
    #         values_fail_noise = np.concatenate((densities_noisy_z[inds_fail].reshape((1, -1)), uncerts_noisy_z[inds_fail].reshape((1, -1))),  axis=0).transpose([1, 0])
    #         # values_fail_neg = np.concatenate((values_fail_normal, values_fail_noise))
    #         values_fail_neg = values_fail_normal
    #         values_fail = np.concatenate((values_fail_neg, values_fail_pos))
    #         # labels_fail = np.concatenate((np.zeros(len(inds_fail)*2), np.ones(len(inds_fail))))
    #         labels_fail = np.concatenate((np.zeros(len(inds_fail)), np.ones(len(inds_fail))))
    #         probs_fail = lr.predict_proba(values_fail)[:, 1]
    #         y_label_pred_fail = lr.predict(values_fail)

    #         accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(labels_fail, y_label_pred_fail)
    #         fprs_fail, tprs_fail, thresholds_fail = roc_curve(labels_fail, probs_fail)
    #         roc_auc_fail = auc(fprs_fail, tprs_fail)

    #         curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
    #                 'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
    #                 'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
    #         results_all.append(curr_result)

    #     print('trained on {} attack and tested on {}- accuracy of pretrained model: {:7.2f}% \
    #     - detection rates ------ SAEs: {:7.2f}%, FAEs: {:7.2f}%'.format(args.attack, attack, 100*acc_suc, 100*tpr_success, 100*tpr_fail))

    #     import csv
    #     with open('{}{}_train_{}_test_{}.csv'.format(kd_bu_results_dir, args.dataset, args.attack, attack), 'w', newline='') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for row in results_all:
    #             writer.writerow(row)
            
    # print('Done!')

    #For gray-box attacks
    if not(args.attack=='hop' or args.attack=='sa' or args.attack=='sta' or (args.attack=='df' and args.dataset=='tiny')):
        for attack in [args.attack]:
            results_all = []
            #Prepare data
            # Load adversarial samples
            adv_path = '{}{}_{}.npy'.format(adv_data_gray_dir, args.dataset, attack)
            X_test_adv = np.load(adv_path)
            X_test_adv = X_test_adv[inds_correct]
            X_test_noisy = get_noisy_samples(X_test, X_test_adv, args.dataset, attack)

            uncerts_adv_file = '{}{}_{}_uncerts_adv.npy'.format(kd_bu_results_gray_dir, args.dataset, attack)
            if os.path.isfile(uncerts_adv_file):
                uncerts_adv = np.load(uncerts_adv_file)
            else:
                uncerts_adv = get_mc_predictions(model, X_test_adv, batch_size=args.batch_size).var(axis=0).mean(axis=1)
                np.save(uncerts_adv_file, uncerts_adv)

            X_test_adv_features_file = '{}{}_{}_dens_adv.npy'.format(kd_bu_results_gray_dir, args.dataset, attack)
            if os.path.isfile(X_test_adv_features_file):
                X_test_adv_features = np.load(X_test_adv_features_file)
            else:
                X_test_adv_features = get_deep_representations(model, X_test_adv, batch_size=args.batch_size, dataset=args.dataset)
                np.save(X_test_adv_features_file, X_test_adv_features)
            
            # Get model predictions
            preds_test_adv = model.predict(X_test_adv)
            loss, acc_suc = model.evaluate(X_test_adv, Y_test, verbose=0)
            preds_test_adv = preds_test_adv.argmax(axis=1)
            inds_success = np.where(preds_test_adv != Y_test.argmax(axis=1))[0]
            inds_fail = np.where(preds_test_adv == Y_test.argmax(axis=1))[0]

            # Get density estimates
            densities_adv = score_samples(kdes, X_test_adv_features, preds_test_adv)

            # Z-score the uncertainty and density values
            uncerts_adv_z = uncerts_scaler.transform(uncerts_adv.reshape((-1,1))).reshape((-1,))
            densities_adv_z = dense_scaler.transform(densities_adv.reshape((-1,1))).reshape((-1,))

            #Predict all
            values_pos = np.concatenate((densities_adv_z.reshape((1, -1)), uncerts_adv_z.reshape((1, -1))),  axis=0).transpose([1, 0])
            values_normal = np.concatenate((densities_normal_z.reshape((1, -1)), uncerts_normal_z.reshape((1, -1))),  axis=0).transpose([1, 0])
            values_noise = np.concatenate((densities_noisy_z.reshape((1, -1)), uncerts_noisy_z.reshape((1, -1))),  axis=0).transpose([1, 0])
            values_neg = values_normal
            values = np.concatenate((values_neg, values_pos))
            labels = np.concatenate((np.zeros(len(values_normal)), np.ones(len(values_pos))))
            probs = lr.predict_proba(values)[:, 1]
            y_label_pred = lr.predict(values)

            acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all= evalulate_detection_test(labels, y_label_pred)
            fprs_all, tprs_all, thresholds_all = roc_curve(labels, probs)
            roc_auc_all = auc(fprs_all, tprs_all)
            print("AUC: {:.4f}%, Overall accuracy: {:.4f}%, FPR value: {:.4f}%".format(100*roc_auc_all, 100*acc_all, 100*fpr_all))

            curr_result = {'type':'all', 'nsamples': len(inds_correct),	'acc_suc': acc_suc,	\
                            'acc': acc_all, 'tpr': tpr_all, 'fpr': fpr_all, 'tp': tp_all, 'ap': ap_all, 'fb': fb_all, 'an': an_all,	\
                            'tprs': list(fprs_all), 'fprs': list(tprs_all),	'auc': roc_auc_all}
            results_all.append(curr_result)

            #Predict Success
            if len(inds_success)==0:
                tpr_success=np.nan
                curr_result = {'type':'success', 'nsamples': 0,	'acc_suc': 0,	\
                        'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                        'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
                results_all.append(curr_result)
            else:
                values_success_pos = np.concatenate((densities_adv_z[inds_success].reshape((1, -1)), uncerts_adv_z[inds_success].reshape((1, -1))),  axis=0).transpose([1, 0])
                values_success_normal = np.concatenate((densities_normal_z[inds_success].reshape((1, -1)), uncerts_normal_z[inds_success].reshape((1, -1))),  axis=0).transpose([1, 0])
                values_success_noise = np.concatenate((densities_noisy_z[inds_success].reshape((1, -1)), uncerts_noisy_z[inds_success].reshape((1, -1))),  axis=0).transpose([1, 0])
                # values_success_neg = np.concatenate((values_success_normal, values_success_noise))
                values_success_neg = values_success_normal
                values_success = np.concatenate((values_success_neg, values_success_pos))
                # labels_success = np.concatenate((np.zeros(len(inds_success)*2), np.ones(len(inds_success))))
                labels_success = np.concatenate((np.zeros(len(inds_success)), np.ones(len(inds_success))))
                probs_success = lr.predict_proba(values_success)[:, 1]
                y_label_pred_success = lr.predict(values_success)

                accuracy_success, tpr_success, fpr_success, tp_success, ap_success, fb_success, an_success = evalulate_detection_test(labels_success, y_label_pred_success)
                fprs_success, tprs_success, thresholds_success = roc_curve(labels_success, probs_success)
                roc_auc_success = auc(fprs_success, tprs_success)

                curr_result = {'type':'success', 'nsamples': len(inds_success),	'acc_suc': 0,	\
                        'acc': accuracy_success, 'tpr': tpr_success, 'fpr': fpr_success, 'tp': tp_success, 'ap': ap_success, 'fb': fb_success, 'an': an_success,	\
                        'tprs': list(fprs_success), 'fprs': list(tprs_success),	'auc': roc_auc_success}
                results_all.append(curr_result)
            
            #Predict Fail
            if len(inds_fail)==0:
                tpr_fail=np.nan
                curr_result = {'type':'fail', 'nsamples': 0,	'acc_suc': 0,	\
                        'acc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tp': np.nan, 'ap': np.nan, 'fb': np.nan, 'an': np.nan,	\
                        'tprs': np.nan, 'fprs': np.nan,	'auc': np.nan}
                results_all.append(curr_result)
            else:
                values_fail_pos = np.concatenate((densities_adv_z[inds_fail].reshape((1, -1)), uncerts_adv_z[inds_fail].reshape((1, -1))),  axis=0).transpose([1, 0])
                values_fail_normal = np.concatenate((densities_normal_z[inds_fail].reshape((1, -1)), uncerts_normal_z[inds_fail].reshape((1, -1))),  axis=0).transpose([1, 0])
                values_fail_noise = np.concatenate((densities_noisy_z[inds_fail].reshape((1, -1)), uncerts_noisy_z[inds_fail].reshape((1, -1))),  axis=0).transpose([1, 0])
                # values_fail_neg = np.concatenate((values_fail_normal, values_fail_noise))
                values_fail_neg = values_fail_normal
                values_fail = np.concatenate((values_fail_neg, values_fail_pos))
                # labels_fail = np.concatenate((np.zeros(len(inds_fail)*2), np.ones(len(inds_fail))))
                labels_fail = np.concatenate((np.zeros(len(inds_fail)), np.ones(len(inds_fail))))
                probs_fail = lr.predict_proba(values_fail)[:, 1]
                y_label_pred_fail = lr.predict(values_fail)

                accuracy_fail, tpr_fail, fpr_fail, tp_fail, ap_fail, fb_fail, an_fail = evalulate_detection_test(labels_fail, y_label_pred_fail)
                fprs_fail, tprs_fail, thresholds_fail = roc_curve(labels_fail, probs_fail)
                roc_auc_fail = auc(fprs_fail, tprs_fail)

                curr_result = {'type':'fail', 'nsamples': len(inds_fail),	'acc_suc': 0,	\
                        'acc': accuracy_fail, 'tpr': tpr_fail, 'fpr': fpr_fail, 'tp': tp_fail, 'ap': ap_fail, 'fb': fb_fail, 'an': an_fail,	\
                        'tprs': list(fprs_fail), 'fprs': list(tprs_fail),	'auc': roc_auc_fail}
                results_all.append(curr_result)

            print('Gray-box attack {}- accuracy of pretrained model: {:7.2f}% \
            - detection rates ------ SAEs: {:7.2f}%, FAEs: {:7.2f}%'.format(attack, 100*acc_suc, 100*tpr_success, 100*tpr_fail))

            import csv
            with open('{}{}_gray_{}.csv'.format(kd_bu_results_gray_dir, args.dataset, args.attack), 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results_all:
                    writer.writerow(row)
                
        print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either {}".format(DATASETS),
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either {}".format(ATTACK),
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=256)
    args = parser.parse_args()
    main(args)
