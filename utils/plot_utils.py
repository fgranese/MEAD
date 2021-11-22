import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm


def plot_font_setup(ax_plt, legend_title, x_label, y_label, tick_size=4, legend_title_size=15, legend_content_size=15,
                    label_size=7):
    ax_plt.tick_params(axis='both', which='both', labelsize=tick_size)
    h, l = ax_plt.get_legend_handles_labels()
    ax_plt.legend(h, l, title=legend_title, prop={'size': legend_content_size}, title_fontsize=legend_title_size)
    plt.ylabel(y_label, fontsize=label_size)
    plt.xlabel(x_label, fontsize=label_size)


def plot_rocs(T_0, T_1, labels, colors, plot_name):
    """
    Args:
        T_0: list of arrays for x axis (one for each curve)
        T_1: list of arrays for y axis (one for each curve)
        labels: list of strings for label names (one for each curve)
        colors: list of strings of colour (one for each curve)
        loss_train: name of the loss used for training the detector
        plot_name: name of the file to save the plot
    """

    fig, ax1 = plt.subplots(nrows=1, ncols=1, dpi=170)
    plot_font_setup(ax_plt=ax1, legend_title='', x_label='', y_label='')

    for i in range(len(T_0)):
        ax1.plot(T_0[i], T_1[i], label=labels[i], color=colors[i])
        x = np.interp(0.95, np.sort(T_1[i]), np.sort(T_0[i]))
        print(labels[i], "AUROC", round(skm.auc(np.sort(T_0[i]), np.sort(T_1[i])) * 100, 1), "FPR", round(x * 100, 1))
        ax1.scatter(x=x, y=0.95, color=colors[i], marker='x')

    for ax in [ax1]:
        ax.set(xlabel='FPR', ylabel='TPR')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.legend(title='', prop={'size': 7}, title_fontsize=5)

    plt.subplots_adjust(bottom=0.15, wspace=0.5)
    plt.grid(b=True, which='major', color='#666666', linestyle='--', alpha=0.2)
    plt.axhline(y=0.95, color='red', linestyle=':')

    plt.savefig(plot_name)

    plt.tight_layout()
    plt.show()

