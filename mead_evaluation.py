from utils.plot_utils import plot_rocs
from utils.general_utils import load_data, load_model, get_prediction_by_bs
from setup_paths import *

import numpy as np
import torch

import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_statistics(dataset, device, X_adv):

    model = load_model(dataset, checkpoints_dir, device)
    model.eval()
    x_train, y_train, x_test, y_test = load_data(args.dataset)

    if torch.cuda.is_available():
        preds_adv = get_prediction_by_bs(X=X_adv, model=model, num_classes=10)
    else:
        preds_adv = model(torch.tensor(X_adv))

    correct = np.where(preds_adv.argmax(axis=1) != y_test.argmax(axis=1), 1, 0)

    return correct

def collect_decision_by_thresholds(probs, thrs_size):
    thrs = np.linspace(probs.min(), probs.max(), thrs_size)
    decision_by_thr = np.zeros((len(probs), thrs_size))

    # An example is detected as adversarial is the prediction is above the threshold, natural if not
    for i in range(thrs_size):
        thr = thrs[i]
        y_pred = np.where(probs > thr, 1, 0)
        decision_by_thr[:, i] = y_pred

    return decision_by_thr


def collect_decision(args, method_name, thrs_size=200):
    correct = []

    for i in range(len(ATTACKS)):
        attack = ATTACKS[i]
        print(attack)

        # Download Adversarial Samples
        X_test_adv = np.load('%s%s/%s%s.npy' % (adv_data_dir, args.dataset, args.dataset, attack))

        # Compute whether the adversarial examples successfully fools the target classifier or not, and save the decision
        correct.append(get_statistics(dataset=args.dataset, X_adv=X_test_adv, device=args.device))
        # Download whether this is an adversarial or a natural sample
        labels = np.load('{}{}/{}/evaluation/labels_{}{}_all.npy'.format(features_dir, method_name, args.dataset, args.dataset, attack))

        # Download the detector's output (distance or probability)
        probs = np.load('{}{}/{}/evaluation/probs_{}{}_all.npy'.format(features_dir, method_name, args.dataset, args.dataset, attack))

        # For FS and MagNet, it outputs uniquely one distance. For the others, it outputs 2 : one from the point of view of the natural decision (i.e. the value is big is the samples is susceptible to be natural), and one from the point of view of the adversarial decision (i.e. the value is big if the sample is susceptible to be adversarial)
        if method_name == 'fs':
            proba = probs
        elif method_name == 'magnet':
            if probs.shape[0] > 1:
                proba = probs[1]
            else:
                proba = probs[0]
        # LID actually label the natural examples as 1 and adversarial one as 0, so we have to reverse it.
        elif method_name == 'lid':
            labels = 1 - labels
            proba = probs[:, 1]
            proba = 1 - proba
        else:
            proba = probs[:, 1]

        if i == 0:
            # We reshape the variable that if the noisy sample is successful or not
            ca = np.ones((2 * len(X_test_adv), thrs_size))
            for j in range(thrs_size):
                ca[len(X_test_adv):, j] = np.asarray(correct)
            # We multiply the adversarial decision by the successfulness of the attack to discard the non-adversarial samples.
            decision = collect_decision_by_thresholds(proba, thrs_size)
            decision_by_thr_adv = decision * ca
            decision_by_thr_nat = decision
        else:
            # We gather the decisions for all the attacks
            ca = np.ones((2 * len(X_test_adv), thrs_size))
            for j in range(thrs_size):
                ca[len(X_test_adv):, j] = np.asarray(correct)[i, :]
            decision = collect_decision_by_thresholds(proba, thrs_size)
            decision_by_thr_adv = decision_by_thr_adv + decision * ca
            decision_by_thr_nat = decision_by_thr_nat + decision

    correct = np.transpose(correct)

    # We compute the statistics of the successfulness of the attacks
    mean_success_adv_per_sample = correct.mean(1)
    mean_success_adv = mean_success_adv_per_sample.mean(0)

    print('Avg. Number of Successful Attacks per Natural Sample: ', mean_success_adv * len(ATTACKS))
    print('Total. Number of Successful Attacks per Natural Sample: ', len(ATTACKS))

    # We gather the true label (i.e. 0 if the sample is natural, 1 if it is not).
    labels_tot = (np.ones((thrs_size, len(labels))) * labels).transpose()

    correct = np.concatenate((np.zeros(correct.shape), correct), axis=0)
    correct_all = np.zeros(decision_by_thr_adv.shape)
    # We compute the number of times a natural sample has a successful adversarial examples.
    for i in range(decision_by_thr_adv.shape[1]):
        correct_all[:, i] = correct.sum(1)

    # The sample is considered as true positive iff there is at least one successful adversarial examples (i.e. correct_all > 0) and iff the detector detects all of successful adversarial examples (i.e. decision_by_thr_adv = correct_all).
    tp = np.where((decision_by_thr_adv == correct_all) & (correct_all > 0), 1, 0)
    # The sample is considered as false positive if it was a natural example (i.e. labels_tot==0) and if it detected as adversarial (the decision is above 0).
    fp = np.where((decision_by_thr_nat > 0) & (labels_tot == 0), 1, 0)
    # The sample is considered as false positive if it was a natural example (i.e. labels_tot==0) and if it detected as natural (the natural decision is 0).
    tn = np.where((decision_by_thr_nat == 0) & (labels_tot == 0), 1, 0)
    # The sample is considered as false negative iff here is at least one successful adversarial examples (i.e. correct_all > 0) and if it detected less examples than there is (decision_by_thr_adv < correct_all).
    fn = np.where((decision_by_thr_adv < correct_all) & (correct_all > 0), 1, 0)

    # We sum over all the examples.
    tpr = tp.sum(axis=0) / (tp.sum(axis=0) + fn.sum(axis=0))
    fpr = fp.sum(axis=0) / (fp.sum(axis=0) + tn.sum(axis=0))
    # We plot the roc and print the AUROC value
    if not os.path.exists(results_dir + method_name):
        os.makedirs(results_dir + method_name)
    plot_rocs([fpr], [tpr], ["MEAD"], ['red'], '{}{}/{}'.format(results_dir, method_name, args.plot_name))

    return fpr, tpr

def main(args, method_name):
    collect_decision(args, method_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either {}".format(DATASETS),
        required=True, type=str
    )
    parser.add_argument(
        '-m', '--method_name',
        help="Dataset to use; either {}".format(DATASETS),
        required=True, type=str
    )
    parser.add_argument(
        '-pn', '--plot_name',
        help="Plot name", required=True
    )
    parser.add_argument(
        '-dev', '--device',
        help="cuda/cpu", required=True
    )

    args = parser.parse_args()
    main(args, args.method_name)