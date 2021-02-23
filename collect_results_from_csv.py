import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from common.util import *
from setup_paths import *
import csv
csv.field_size_limit(sys.maxsize)


CSVs_dir=[
    kd_bu_results_dir,
    lid_results_dir,
    nss_results_dir,
    fs_results_dir,
    magnet_results_dir,
    dnr_results_dir,
    sfad_results_dir,
    nic_results_dir
]

CSVs_gray_dir=[
    kd_bu_results_gray_dir,
    lid_results_gray_dir,
    nss_results_gray_dir,
    fs_results_gray_dir,
    magnet_results_gray_dir,
    dnr_results_gray_dir,
    sfad_results_gray_dir,
    nic_results_gray_dir
]

fn = ['Attack', 'KD+BU_DR', 'KD+BU_FPR', 'LID_DR', 'LID_FPR', 'NSS_DR', 'NSS_FPR', \
            'FS_DR', 'FS_FPR', 'MagNet_DR', 'MagNet_FPR', 'DNR_DR', 'DNR_FPR', \
            'SFAD_DR', 'SFAD_FPR', 'NIC_DR', 'NIC_FPR']

fn_g = ['Attack', 'KD+BU_DR', 'LID_DR', 'NSS_DR',  \
            'FS_DR', 'MagNet_DR', 'DNR_DR', \
            'SFAD_DR', 'NIC_DR']

for ds in DATASETS:
    current_ds_s_csv_file = '{}detectors_s_{}.csv'.format(results_path, ds)
    current_ds_f_csv_file = '{}detectors_f_{}.csv'.format(results_path, ds)

    s_dict = [{} for _ in range(len(ALL_ATTACKS))]
    f_dict = [{} for _ in range(len(ALL_ATTACKS))]

    ATTACKS=ATTACK[DATASETS.index(ds)]
    for atk in ATTACKS:
        att_indx = ALL_ATTACKS.index(atk)

        s = {'Attack':atk, 'KD+BU_DR': '-', 'KD+BU_FPR': '-', 'LID_DR': '-', 'LID_FPR': '-', 'NSS_DR': '-', 'NSS_FPR': '-', \
            'FS_DR': '-', 'FS_FPR': '-', 'MagNet_DR': '-', 'MagNet_FPR': '-', 'DNR_DR': '-', 'DNR_FPR': '-', \
            'SFAD_DR': '-', 'SFAD_FPR': '-', 'NIC_DR': '-', 'NIC_FPR': '-'}
        f = {'Attack':atk, 'KD+BU_DR': '-', 'KD+BU_FPR': '-', 'LID_DR': '-', 'LID_FPR': '-', 'NSS_DR': '-', 'NSS_FPR': '-', \
            'FS_DR': '-', 'FS_FPR': '-', 'MagNet_DR': '-', 'MagNet_FPR': '-', 'DNR_DR': '-', 'DNR_FPR': '-', \
            'SFAD_DR': '-', 'SFAD_FPR': '-', 'NIC_DR': '-', 'NIC_FPR': '-'}

        for csv_dir in CSVs_dir:
            csv_dir_indx = CSVs_dir.index(csv_dir)*2
            current_result = []
            csv_file = '{}{}_{}.csv'.format(csv_dir, ds, atk)
            if os.path.isfile(csv_file):
                with open(csv_file, 'r') as file: 
                    data = csv.DictReader(file)
                    for row in data:
                        current_result.append(row)
                    
                    FPR=np.round(100*np.float(current_result[0]['fpr']), decimals=2)
                    DRS=np.round(100*np.float(current_result[1]['tpr']), decimals=2)
                    DRF=np.round(100*np.float(current_result[2]['tpr']), decimals=2)

                    key_dr=fn[csv_dir_indx+1]
                    key_fpr=fn[csv_dir_indx+2]
                    
                    s[key_dr] = DRS
                    s[key_fpr] = FPR
                    f[key_dr] = DRF
                    f[key_fpr] = FPR
            
        s_dict[att_indx] = s
        f_dict[att_indx] = f
    
    with open(current_ds_s_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fn)
        writer.writeheader()
        for row in s_dict:
            writer.writerow(row)
    
    with open(current_ds_f_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fn)
        writer.writeheader()
        for row in f_dict:
            writer.writerow(row)
    
for ds in DATASETS:
    current_ds_s_csv_file = '{}detectors_gray_s_{}.csv'.format(results_path, ds)
    current_ds_f_csv_file = '{}detectors_gray_f_{}.csv'.format(results_path, ds)

    s_dict = [{} for _ in range(len(ALL_ATTACKS))]
    f_dict = [{} for _ in range(len(ALL_ATTACKS))]

    ATTACKS=ATTACK_GRAY[DATASETS.index(ds)]
    for atk in ATTACKS:
        att_indx = ALL_ATTACKS.index(atk)

        s = {'Attack':atk, 'KD+BU_DR': '-', 'LID_DR': '-', 'NSS_DR': '-',\
            'FS_DR': '-', 'MagNet_DR': '-', 'DNR_DR': '-', \
            'SFAD_DR': '-', 'NIC_DR': '-'}
        f = {'Attack':atk, 'KD+BU_DR': '-', 'LID_DR': '-', 'NSS_DR': '-',\
            'FS_DR': '-', 'MagNet_DR': '-', 'DNR_DR': '-', \
            'SFAD_DR': '-', 'NIC_DR': '-'}

        for csv_dir in CSVs_gray_dir:
            csv_dir_indx = CSVs_gray_dir.index(csv_dir)
            current_result = []
            csv_file = '{}{}_{}.csv'.format(csv_dir, ds+'_gray', atk)
            if os.path.isfile(csv_file):
                with open(csv_file, 'r') as file: 
                    data = csv.DictReader(file)
                    for row in data:
                        current_result.append(row)
                    
                    # FPR=np.round(100*np.float(current_result[0]['fpr']), decimals=2)
                    DRS=np.round(100*np.float(current_result[1]['tpr']), decimals=2)
                    DRF=np.round(100*np.float(current_result[2]['tpr']), decimals=2)

                    key_dr=fn_g[csv_dir_indx+1]
                    # key_fpr=fn[csv_dir_indx+2]
                    
                    s[key_dr] = DRS
                    # s[key_fpr] = FPR
                    f[key_dr] = DRF
                    # f[key_fpr] = FPR
            
            else:
                csv_file = '{}{}_{}.csv'.format(csv_dir, ds, atk)
                if os.path.isfile(csv_file):
                    with open(csv_file, 'r') as file: 
                        data = csv.DictReader(file)
                        for row in data:
                            current_result.append(row)
                        
                        # FPR=np.round(100*np.float(current_result[0]['fpr']), decimals=2)
                        DRS=np.round(100*np.float(current_result[1]['tpr']), decimals=2)
                        DRF=np.round(100*np.float(current_result[2]['tpr']), decimals=2)

                        key_dr=fn_g[csv_dir_indx+1]
                        # key_fpr=fn[csv_dir_indx+2]
                        
                        s[key_dr] = DRS
                        # s[key_fpr] = FPR
                        f[key_dr] = DRF
                        # f[key_fpr] = FPR
                
            
        s_dict[att_indx] = s
        f_dict[att_indx] = f
    
    with open(current_ds_s_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fn_g)
        writer.writeheader()
        for row in s_dict:
            writer.writerow(row)
    
    with open(current_ds_f_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fn_g)
        writer.writeheader()
        for row in f_dict:
            writer.writerow(row) 

print('Done!')