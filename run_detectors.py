import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle
from common.util import *
from setup_paths import *

# #run KD_BU
# for dataset in ['tiny']:
#     ATTACKS = ATTACK[DATASETS.index(dataset)]
#     for attack in ATTACKS:
#         print("dataset :: {}  -- attack :: {} ".format(dataset, attack))
#         os.system('{}{}detect_kd_bu.py -d={} -a={}'.format(env_param, detectors_dir, dataset, attack))

# #run LID
# for dataset in DATASETS:
#     ATTACKS = ATTACK[DATASETS.index(dataset)]
#     for attack in ATTACKS:
#         print("dataset :: {}  -- attack :: {} ".format(dataset, attack))
#         os.system('{}{}detect_lid.py -d={} -a={} -k={}'.format(env_param, detectors_dir, dataset, attack, k_nn[DATASETS.index(dataset)]))

# #run MagNet
# for dataset in DATASETS:
#     print("dataset :: {}  -- attack :: all ".format(dataset))
#     os.system('{}{}detect_magnet.py -d={}'.format(env_param, detectors_dir, dataset))

# #run FS
# for dataset in DATASETS:
#     print("dataset :: {}  -- attack :: all ".format(dataset))
#     os.system('{}{}detect_fs.py -d={}'.format(env_param, detectors_dir, dataset))

# #run DNR
# for dataset in DATASETS[0:3]:
#     print("dataset :: {}  -- attack :: all ".format(dataset))
#     os.system('{}{}detect_dnr.py -d={}'.format(env_param, detectors_dir, dataset))

# #run NSS
# for dataset in DATASETS:
#     print("dataset :: {}  -- attack :: all ".format(dataset))
#     os.system('{}{}detect_nss.py -d={}'.format(env_param, detectors_dir, dataset))


# #run SFAD
# for dataset in DATASETS:
#     print("dataset :: {}  -- attack :: all ".format(dataset))
#     os.system('{}{}detect_sfad.py -d={}'.format(env_param, detectors_dir, dataset))

#run NIC
for dataset in DATASETS:
    print("dataset :: {}  -- attack :: all ".format(dataset))
    os.system('{}{}detect_nic.py -d={}'.format(env_param, detectors_dir, dataset))