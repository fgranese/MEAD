import os

# -------------------------- Common
checkpoints_dir = 'checkpoints/'
adv_data_dir = 'adv_data/'
results_dir = 'results/'
features_dir = 'features/'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

DATASETS = ['mnist', 'cifar10']

# All the losses considered
#LOSS = ['CE', 'KL', 'Rao', 'g']

LOSS = ['CE']
ATTACK = ['fgsm_0.03125', 'bim_0.03125', 'pgdi_0.03125']

# All the attacks considered
# ATTACK = [
#     'fgsm_0.03125', 'fgsm_0.0625', 'fgsm_0.125', 'fgsm_0.25', 'fgsm_0.3125', 'fgsm_0.5',
#     'bim_0.03125', 'bim_0.0625', 'bim_0.125', 'bim_0.25', 'bim_0.3125', 'bim_0.5',
#     'pgdi_0.03125', 'pgdi_0.0625', 'pgdi_0.125', 'pgdi_0.25', 'pgdi_0.3125', 'pgdi_0.5',
#     'pgd1_5', 'pgd1_10', 'pgd1_15', 'pgd1_20', 'pgd1_25', 'pgd1_30', 'pgd1_40',
#     'pgd2_0.125', 'pgd2_0.25', 'pgd2_0.3125', 'pgd2_0.5', 'pgd2_1', 'pgd2_1.5', 'pgd2_2',
#     'cwi', 'df', 'sa', 'hop', 'sta', 'cw2'
# ]


ATTACKS = []
for i in range(len(ATTACK)):
    if ATTACK[i].startswith('fgsm') or ATTACK[i].startswith('pgd') or ATTACK[i].startswith('bim'):
        for loss in LOSS:
            ATTACKS += ["{}_{}".format(loss, ATTACK[i])]
    else:
        ATTACKS += ["_{}".format(ATTACK[i])]

fieldnames = ['type', 'nsamples', 'acc_suc', 'acc', 'tpr', 'fpr', 'tp', 'ap', 'fb', 'an', 'tprs', 'fprs', 'auc']


# -------------------------- detect NSS
pgd_percent = [[0.02, 0.1, 0.18, 0.3, 0.3, 0.1],
               [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]]

# -------------------------- detect KD_BU
# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar10': 0.1}

# -------------------------- detect MagNet


