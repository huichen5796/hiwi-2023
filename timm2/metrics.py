import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import os

def ece_score(targets, outputs, classes, classes_to_idx, num_bins=31, plot_ece=False, save_path=None):

    ECE = []
    for c in classes:
        it = classes_to_idx[c]
        confidence = outputs[...,it]
        #confidence = np.sort(confidence)
        predictions = (confidence>0.5).astype(int)
        tt = targets[...,it]
        tp = (predictions*tt).astype(int)
        tn = ((1-predictions)*(1-tt)).astype(int)

        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(tt, confidence, n_bins=num_bins, strategy='quantile')
        ece = np.mean(np.abs(prob_true-prob_pred))
        ECE.append(ece)

        if plot_ece:
            def plot_conf(acc, conf):
                plt.rcParams['text.usetex'] = True
                plt.rcParams["savefig.dpi"] = 600

                fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.25))
                ax.plot([0, 1], [0, 1], 'k--')
                ax.plot(conf, acc, marker='.')
                ax.set_xlabel(r'confidence')
                ax.set_ylabel(r'accuracy')
                ax.set_xticks((np.arange(0, 1.1, step=0.2)))
                ax.set_yticks((np.arange(0, 1.1, step=0.2)))

                return fig, ax

            fig_freq, ax_freq = plot_conf(prob_true, prob_pred)

            textstr_freq_uc = r'ECE\,=\,{:.2f}'.format(ece)
            props = dict(boxstyle='round', facecolor='white', alpha=0.75)
            ax_freq.text(0.075, 0.925, textstr_freq_uc, transform=ax_freq.transAxes, fontsize=14,
                            verticalalignment='top',
                            horizontalalignment='left',
                            bbox=props
                            )
            ax_freq.set_title(r'{}'.format(c))
            fig_freq.tight_layout()
            if save_path:
                filename = f'reliability_{c}.png'
                fig_freq.savefig(os.path.join(save_path, filename), format='png')
            else:
                fig_freq.show()



        plt.clf()
        plt.close()
        np.asarray(ECE, dtype=float)
    return np.mean(ECE), ECE


def roc_auc_score(targets, outputs):
    auc_roc = metrics.roc_auc_score(targets, outputs, average=None).astype(float)
    return np.mean(auc_roc), auc_roc

def roc_pr_score(targets, outputs):
    auc_pr = metrics.average_precision_score(targets, outputs, average=None).astype(float)
    return np.mean(auc_pr), auc_pr
