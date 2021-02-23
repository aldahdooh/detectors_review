from __future__ import division, absolute_import, print_function
import argparse
from common.util import *
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn', 'tiny', 'tiny_gray'], \
        "dataset parameter must be either 'mnist', 'cifar_cnn', 'cifar_densenet', 'svhn', or 'tiny'"
    print('Data set: %s' % args.dataset)
    
    if args.dataset == 'mnist':
        from baselineCNN.cnn.cnn_mnist import MNISTCNN as model
        model_mnist = model(mode='train', filename='cnn_{}.h5'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    elif args.dataset == 'mnist_gray':
        from baselineCNN.cnn.cnn_mnist_gray import MNISTCNN as model
        model_mnist = model(mode='train', filename='cnn_{}.h5'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    elif args.dataset == 'cifar':
        from baselineCNN.cnn.cnn_cifar10 import CIFAR10CNN as model
        model_cifar = model(mode='train', filename='cnn_{}.h5'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    elif args.dataset == 'cifar_gray':
        from baselineCNN.cnn.cnn_cifar10_gray import CIFAR10CNN as model
        model_cifar = model(mode='train', filename='cnn_{}.h5'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    elif args.dataset == 'svhn':
        from baselineCNN.cnn.cnn_svhn import SVHNCNN as model
        model_svhn = model(mode='train', filename='cnn_{}.h5'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    elif args.dataset == 'svhn_gray':
        from baselineCNN.cnn.cnn_svhn_gray import SVHNCNN as model
        model_svhn = model(mode='train', filename='cnn_{}.h5'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    elif args.dataset == 'tiny':
        from baselineCNN.cnn.cnn_tiny import TINYCNN as model
        model_tiny = model(mode='train', filename='cnn_{}.h5'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    elif args.dataset == 'tiny_gray':
        from baselineCNN.cnn.cnn_tiny_gray import TINYCNN as model
        model_tiny = model(mode='train', filename='cnn_{}.h5'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar', 'svhn', or 'tiny'",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )

    parser.set_defaults(epochs=50)
    parser.set_defaults(batch_size=128)
    args = parser.parse_args()
    main(args)
