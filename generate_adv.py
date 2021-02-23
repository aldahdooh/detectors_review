from __future__ import division, absolute_import, print_function
import argparse

from tensorflow.python.keras.backend_config import epsilon
from common.util import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, CarliniL2Method, CarliniLInfMethod, ProjectedGradientDescent, DeepFool, ThresholdAttack, PixelAttack, SpatialTransformation, SquareAttack, ZooAttack, BoundaryAttack, HopSkipJump
from art.classifiers import KerasClassifier

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# def batch(iterable_1, iterable_2, batch_size=10):
#   l = len(iterable_1)
#   for ndx in range(0, l, batch_size):
#       #print(ndx)
#       yield ndx, iterable_1[ndx:min(ndx + batch_size, l)], iterable_2[ndx:min(ndx + batch_size, l)]

def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn', 'tiny', 'tiny_gray'], \
        "dataset parameter must be either 'mnist', 'cifar', 'svhn', or 'tiny'"
    print('Dataset: %s' % args.dataset)
    adv_path = '/home/aaldahdo/detectors/adv_data/'

    if args.dataset == 'mnist':
        from baselineCNN.cnn.cnn_mnist import MNISTCNN as model
        model_mnist = model(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        classifier=model_mnist.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.3
        pa_th=78
        # random_restart = 20
        # x_train = model_mnist.x_train
        x_test = model_mnist.x_test
        # y_train = model_mnist.y_train
        y_test = model_mnist.y_test
        y_test_labels = model_mnist.y_test_labels
        translation = 10
        rotation = 60
    
    elif args.dataset == 'mnist_gray':
        from baselineCNN.cnn.cnn_mnist_gray import MNISTCNN as model
        model_mnist = model(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        classifier=model_mnist.model
        sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.3
        pa_th=78
        # random_restart = 20
        # x_train = model_mnist.x_train
        x_test = model_mnist.x_test
        # y_train = model_mnist.y_train
        y_test = model_mnist.y_test
        y_test_labels = model_mnist.y_test_labels
        translation = 10
        rotation = 60

    elif args.dataset == 'cifar':
        from baselineCNN.cnn.cnn_cifar10 import CIFAR10CNN as model
        model_cifar = model(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        classifier=model_cifar.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        pa_th=100
        # x_train = model_cifar.x_train
        x_test = model_cifar.x_test
        # y_train = model_cifar.y_train
        y_test = model_cifar.y_test
        y_test_labels = model_cifar.y_test_labels
        translation = 8
        rotation = 30
    
    elif args.dataset == 'cifar_gray':
        from baselineCNN.cnn.cnn_cifar10_gray import CIFAR10CNN as model
        model_cifar = model(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        classifier=model_cifar.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        pa_th=100
        # x_train = model_cifar.x_train
        x_test = model_cifar.x_test
        # y_train = model_cifar.y_train
        y_test = model_cifar.y_test
        y_test_labels = model_cifar.y_test_labels
        translation = 8
        rotation = 30

    elif args.dataset == 'svhn':
        from baselineCNN.cnn.cnn_svhn import SVHNCNN as model
        model_svhn = model(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        classifier=model_svhn.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        pa_th=100
        # x_train = model_svhn.x_train
        x_test = model_svhn.x_test
        # y_train = model_svhn.y_train
        y_test = model_svhn.y_test
        y_test_labels = model_svhn.y_test_labels
        translation = 10
        rotation = 60

    elif args.dataset == 'svhn_gray':
        from baselineCNN.cnn.cnn_svhn_gray import SVHNCNN as model
        model_svhn = model(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        classifier=model_svhn.model
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        pa_th=100
        # x_train = model_svhn.x_train
        x_test = model_svhn.x_test
        # y_train = model_svhn.y_train
        y_test = model_svhn.y_test
        y_test_labels = model_svhn.y_test_labels
        translation = 10
        rotation = 60

    elif args.dataset == 'tiny':
        from baselineCNN.cnn.cnn_tiny import TINYCNN as model
        model_tiny = model(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        classifier=model_tiny.model
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        pa_th=100
        # x_train = model_tiny.x_train
        x_test = model_tiny.x_test
        # y_train = model_tiny.y_train
        y_test = model_tiny.y_test
        y_test_labels = model_tiny.y_test_labels
        translation = 8
        rotation = 30
        del model_tiny

    elif args.dataset == 'tiny_gray':
        from baselineCNN.cnn.cnn_tiny_gray import TINYCNN as model
        model_tiny = model(mode='load', filename='cnn_{}.h5'.format(args.dataset))
        classifier=model_tiny.model
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        classifier.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        kclassifier = KerasClassifier(model=classifier, clip_values=(0, 1))
        epsilons=[8/256, 16/256, 32/256, 64/256, 80/256, 128/256]
        epsilons1=[5, 10, 15, 20, 25, 30, 40]
        epsilons2=[0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2]
        eps_sa=0.125
        # x_train = model_tiny.x_train
        x_test = model_tiny.x_test
        # y_train = model_tiny.y_train
        y_test = model_tiny.y_test
        y_test_labels = model_tiny.y_test_labels
        translation = 8
        rotation = 30
        del model_tiny

    
    # batch_count_start = args.batch_indx
    # bsize = args.batch_size
    # batch_count_end = batch_count_start + 1

    #FGSM
    for e in epsilons:
        attack = FastGradientMethod(estimator=kclassifier, eps=e, eps_step=0.01, batch_size=256)
        adv_data = attack.generate(x=x_test)
        adv_file_path = adv_path + args.dataset + '_fgsm_' + str(e) + '.npy'
        np.save(adv_file_path, adv_data)
        print('Done - {}'.format(adv_file_path))
    
    #BIM
    for e in epsilons:
        attack = BasicIterativeMethod(estimator=kclassifier, eps=e, eps_step=0.01, batch_size=32, max_iter=int(e*256*1.25))
        adv_data = attack.generate(x=x_test)
        adv_file_path = adv_path + args.dataset + '_bim_' + str(e) + '.npy'
        np.save(adv_file_path, adv_data)
        print('Done - {}'.format(adv_file_path))
    
    #PGD1
    for e in epsilons1:
        attack = ProjectedGradientDescent(estimator=kclassifier, norm=1, eps=e, eps_step=4, batch_size=32)
        adv_data = attack.generate(x=x_test)
        adv_file_path = adv_path + args.dataset + '_pgd1_' + str(e) + '.npy'
        np.save(adv_file_path, adv_data)
        print('Done - {}'.format(adv_file_path))
    
    #PGD2
    for e in epsilons2:
        attack = ProjectedGradientDescent(estimator=kclassifier, norm=2, eps=e, eps_step=0.1, batch_size=32)
        adv_data = attack.generate(x=x_test)
        adv_file_path = adv_path + args.dataset + '_pgd2_' + str(e) + '.npy'
        np.save(adv_file_path, adv_data)
        print('Done - {}'.format(adv_file_path))
    
    #PGDInf
    for e in epsilons:
        attack = ProjectedGradientDescent(estimator=kclassifier, norm=np.inf, eps=e, eps_step=0.01, batch_size=32)
        adv_data = attack.generate(x=x_test)
        adv_file_path = adv_path + args.dataset + '_pgdi_' + str(e) + '.npy'
        np.save(adv_file_path, adv_data)
        print('Done - {}'.format(adv_file_path))

    #CWi
    attack = CarliniLInfMethod(classifier=kclassifier, max_iter=200)
    adv_data = attack.generate(x=x_test)
    adv_file_path = adv_path + args.dataset + '_cwi.npy'
    np.save(adv_file_path, adv_data)
    print('Done - {}'.format(adv_file_path))

    # #CWi
    # if args.dataset=='tiny':
    #     for n, x, y in batch(x_test, y_test, batch_size=bsize):
    #         if n>=batch_count_start*bsize and n<batch_count_end*bsize:
    #             adv_file_path = adv_path + args.dataset + '_cwi_' + str(batch_count_start) + '.npy'
    #             if not os.path.isfile(adv_file_path):
    #                 attack = CarliniLInfMethod(classifier=kclassifier, max_iter=100, batch_size=bsize)
    #                 adv_data = attack.generate(x=x)
    #                 np.save(adv_file_path, adv_data)
    #                 print('Done - {}'.format(adv_file_path))

    #CW2 - SLOW
    attack = CarliniL2Method(classifier=kclassifier, max_iter=100, batch_size=1, confidence=10)
    adv_data = attack.generate(x=x_test)
    adv_file_path = adv_path + args.dataset + '_cw2.npy'
    np.save(adv_file_path, adv_data)
    print('Done - {}'.format(adv_file_path))

    #DF
    attack = DeepFool(classifier=kclassifier)
    adv_data = attack.generate(x=x_test)
    adv_file_path = adv_path + args.dataset + '_df.npy'
    np.save(adv_file_path, adv_data)
    print('Done - {}'.format(adv_file_path))

    # #DF
    # if args.dataset=='tiny':
    #     for n, x, y in batch(x_test, y_test, batch_size=bsize):
    #         if n>=batch_count_start*bsize and n<batch_count_end*bsize:
    #             attack = DeepFool(classifier=kclassifier, epsilon=9, max_iter=100)
    #             adv_data = attack.generate(x=x)
    #             adv_file_path = adv_path + args.dataset + '_df_'+ str(batch_count_start) + '.npy'
    #             np.save(adv_file_path, adv_data)
    #             print('Done - {}'.format(adv_file_path))

    #Spatial transofrmation attack
    attack = SpatialTransformation(classifier=kclassifier, max_translation=translation, max_rotation=rotation)
    adv_data = attack.generate(x=x_test)
    adv_file_path = adv_path + args.dataset + '_sta.npy'
    np.save(adv_file_path, adv_data)
    print('Done - {}'.format(adv_file_path))

    #Square Attack
    attack = SquareAttack(estimator=kclassifier, max_iter=200, eps=eps_sa)
    adv_data = attack.generate(x=x_test, y=y_test)
    adv_file_path = adv_path + args.dataset + '_sa.npy'
    np.save(adv_file_path, adv_data)
    print('Done - {}'.format(adv_file_path))

    #HopSkipJump Attack
    y_test_next= get_next_class(y_test)
    attack = HopSkipJump(classifier=kclassifier, targeted=False, max_iter=0, max_eval=100, init_eval=10)
    
    iter_step = 10
    adv_data = np.zeros(x_test.shape)
    # adv_data = adv_data[0:25]
    for i in range(4):
        adv_data = attack.generate(x=x_test, x_adv_init=adv_data, resume=True)
        attack.max_iter = iter_step

    # _, acc_normal = classifier.evaluate(x_test[0:25], y_test[0:25])
    # _, acc_adv = classifier.evaluate(adv_data, y_test[0:25])
    # print('Normal accuracy - {}\nAttack accuracy - {}'.format(acc_normal, acc_adv))

    # subcount=1
    # for i in range(0, 25):
    #     plt.subplot(5,5,subcount)
    #     if args.dataset=='mnist':
    #         plt.imshow(adv_data[i][:,:,0])
    #     else:
    #         plt.imshow(adv_data[i][:,:,:])
    #     plt.suptitle(args.dataset+ " sb")
    #     subcount = subcount + 1
    # plt.show()

        adv_file_path = adv_path + args.dataset + '_hop.npy'
        np.save(adv_file_path, adv_data)
        print('Done - {}'.format(adv_file_path))

    #ZOO attack
    attack = ZooAttack(classifier=kclassifier, batch_size=32)
    adv_data = attack.generate(x=x_test, y=y_test)
    adv_file_path = adv_path + args.dataset + '_zoo.npy'
    np.save(adv_file_path, adv_data)
    print('Done - {}'.format(adv_file_path))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar', 'svhn', or 'tiny'",
        required=True, type=str
    )
    parser.add_argument(
        '-i', '--batch_indx',
        help="it is used if you need to generate specific AEs to start with batch indx and to end after one batch only",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="it is used if you need to generate specific AEs to start with batch indx and to end after one batch only",
        required=False, type=int
    )
    parser.add_argument(
        '-g', '--gpu',
        help="GPU Support",
        required=False, type=bool
    )
    parser.set_defaults(gpu=False)
    parser.set_defaults(batch_size=2)
    parser.set_defaults(batch_indx=0)
    args = parser.parse_args()
    main(args)
