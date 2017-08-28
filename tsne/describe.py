import typing
import os

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import constants


def plot_2d_simple(X: np.ndarray, label: typing.Union[list, np.ndarray], title: str, ncol: int=2) -> None:
    """ Generates scatterplot of 2D TSNE embedding, coloured by label """
    assert X.shape[0] == len(label)
    assert type(label) in (list, np.ndarray)
    assert type(X) == np.ndarray
    for i, lu in enumerate(np.unique(label)):
        mask = np.array(label) == lu
        plt.scatter(X[mask, 0], X[mask, 1], s=5, cmap=cm.jet, alpha=.2, label=lu)

    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=ncol, markerscale=4.)

    for lh in legend.legendHandles:
        lh.set_alpha(1)

    plt.tick_params(axis='both', which='both', bottom='off', left='off', labelbottom='off', labelleft='off')
    axes = plt.axes()
    axes.set_facecolor('0.97')
    for location in ['left', 'bottom', 'right', 'top']:
        axes.spines[location].set_color('1.0')
    fn = '{}.png'.format(title.replace(' ', '_').lower())
    path = os.path.join(constants.GRAPHS_FOLDER, fn)
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.close('all')


def plot_cluster_bar(series: pd.Series) -> None:
    series = series.sort_index()
    ax = series.plot('bar')
    ax.grid(True, axis='y', alpha=0.1)
    ax.spines['top'].set_color('1.0')
    ax.spines['right'].set_color('1.0')
    path = os.path.join(constants.GRAPHS_FOLDER, 'cluster_bar.png')
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.close('all')


def plot_cluster_bar_grouped(series: pd.Series, ncol: int = 2) -> None:
    #vcs = df.groupby(group_cn)[values_cn].value_counts().unstack()
    df = series.unstack()
    df.sort_index(inplace=True)
    ax = df.plot(kind='bar')
    ax.grid(True, axis='y', alpha=0.1)
    ax.spines['top'].set_color('1.0')
    ax.spines['right'].set_color('1.0')
    plt.xlabel('')
    plt.xticks(rotation=0)
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=ncol)
    path = os.path.join(constants.GRAPHS_FOLDER, 'cluster_bar_grouped.png')
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.close('all')


def plot_cluster_box_grouped(df:pd.DataFrame, group_cn:str = 'cluster', value_cn:str = None) -> None:
    #vcs = df.groupby(group_cn)[values_cn].value_counts().unstack()
    ax = df.boxplot(value_cn, by=group_cn, grid=False)
    ax.get_figure().suptitle('')
    ax.grid(True, axis='y', alpha=0.1)
    ax.spines['top'].set_color('1.0')
    ax.spines['right'].set_color('1.0')
    #ncol = 2# len(series.index.levels[-1])
    plt.xlabel('')
    plt.xticks(rotation=0)
    plt.title('')
    #legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=ncol)
    fn = 'cluster_box_{}_groupedby_{}.png'.format(value_cn, group_cn)
    path = os.path.join(constants.GRAPHS_FOLDER, fn)
    plt.savefig(path, dpi=400, bbox_inches='tight')
    plt.close('all')
