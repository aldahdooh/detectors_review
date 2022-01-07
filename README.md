# Adversarial detection framework
This repository is the implementation of the paper **`Adversarial Example Detection for DNN Models: A Review and Experimental Comparison`** 

## Citation
```
@article{aldahdooh2022adversarial,
      title={Adversarial Example Detection for DNN Models: A Review and Experimental Comparison}, 
      author={Ahmed Aldahdooh and Wassim Hamidouche and Sid Ahmed Fezza and Olivier Deforges},
      journal={Artificial Intelligence Review},
      year={2022},
      publisher={Springer}
}
```

### Note
Publicly available codes for the detectors that are use in this work are customized and the original repositories/papers can be found in:

 1. [KD_BU](https://github.com/rfeinman/detecting-adversarial-samples)
 2. [LID](https://github.com/xingjunm/lid_adversarial_subspace_detection)
 3. [NSS](https://hal.archives-ouvertes.fr/hal-03003468)
 4. [FS](https://github.com/mzweilin/EvadeML-Zoo)
 5. [MagNet](https://github.com/Trevillie/MagNet)
 6. [DNR](https://arxiv.org/abs/1910.00470)
 7. [SFAD](https://aldahdooh.github.io/SFAD/)
 8. [NIC](https://github.com/RU-System-Software-and-Security/NIC)
 
### Requirement
1. Tested on Python 3.8
2. Keras 2.3.1
3. Tensorflow 2.2
4. thundersvm for GPU-based SVM. [Link](https://thundersvm.readthedocs.io/en/latest/)

### setup_paths
Open `setup_paths.py` and set the paths and other detector-related settings.

##  CNNs and surrogate CNNs Training
Run `train_cnn_base.py -d=<dataset> -e=<nb_epochs> -b=<batch_size>`. Currently,  the supported datatsets are `mnist, cifar, svhn, and tiny`.  `cifar` is for CIFAR-10 dataset, and `tiny` is for Tiny-ImageNet.

## Generate adversarial examples
Run `generate_adv.py -d=<dataset>`. We use ART library. You can easily add, update, or remove adversarial attacks. **DON'T** forget to update the attacks arrays in `setup_paths.py`

## Run all detectorsaccuracies
To run all the detector, just execute `run_detectors.py`. Each detector will generate *csv* file that contains detection accuracy, false/true positive rate, and AUC  for successful, fail, and both (all) adversarial examples.

## Run specific detector
To run a specific detector, execute `detect_<detector_name>.py -d=<dataset> -[other detector-related arguments]`

## Collect results
Execute `collect_results_from_csv.py` to aggregate and summarize all the results in one *csv* file per dataset.

## Steps to add new detector method

### Step-1
Create a folder for your detector and put inside it your code and utils code.
### Step-2
Create a Python file `detect_<detector_name>.py` to run the detector for a specific *dataset*. We recommend to follow the code style we follow in the other detectors files. **DON'T** forget to add lines to generate the *csv* file for the result as we did in the `detect_<detector_name>.py`.
### Step-3
Add detector-related arguments in `setup_paths.py`.
