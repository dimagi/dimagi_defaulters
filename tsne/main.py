import os

import pandas as pd

from . import transform
from . import describe
import constants


def run(
        enc_fn='end_of_case_mature_dummies_10k.pkl',
        meta_fn='end_of_case_mature_original_data_for_cases_10k.pkl',
        saved_tsne_fn='id_tsne.csv',
        from_saved=True
        ) -> None:
    meta = pd.read_pickle(os.path.join(constants.DATA_ROOT, meta_fn))
    enc = pd.read_pickle(os.path.join(constants.DATA_ROOT, enc_fn))
    if from_saved:
        clusters = pd.read_csv(os.path.join(constants.DATA_ROOT, saved_tsne_fn))
        cns = [cn for cn in clusters.columns if 'tsne_' in cn]
        clusters['cluster'] = transform.cluster(clusters[cns])
    else:
        exclude_cns = ['form-*-case-*-@case_id', 'form-*-@name', 'form-*-case-*-@date_modified', '_id']
        tsne_cns = list(set(enc.columns) - set(exclude_cns))
        clusters = transform.dimred_and_cluster(enc[tsne_cns], perplexity=100)
        clusters['_id'] = enc['_id'].values
    clusters = clusters.merge(meta[['_id', constants.CASE_ID_CN]], on='_id', how='left')
    run_plots(clusters, meta)
    return clusters, meta


def run_plots(clusters, meta):
    tsne_cns = [cn for cn in clusters.columns if 'tsne_' in cn]
    title = 'Cluster labels in TSNE 2D scatterplot'
    describe.plot_2d_simple(clusters[tsne_cns].values, clusters['cluster'].values, title, ncol=3)

    hiv_cn = 'form-*-case-*-update-*-hiv_status->-last'
    df = clusters.merge(meta[['_id', hiv_cn]], on='_id', how='left')
    title = 'HIV status in TSNE 2D scatterplot'
    describe.plot_2d_simple(df[tsne_cns].values, df[hiv_cn].values.astype(str), title, ncol=3)

    series = clusters.groupby('cluster')['_id'].count()
    describe.plot_cluster_bar(series)
    
    country_cn = 'form-*-case-*-update-*-country->-last'
    df = clusters.merge(meta[['_id', country_cn]], on='_id', how='left')
    vcs = df.groupby('cluster')[country_cn].value_counts()
    tcs = df.groupby('cluster')[country_cn].count()
    describe.plot_cluster_bar_grouped(100 * vcs / tcs, ncol=3)

    count_cn = 'form_count->-current'
    df = clusters.merge(meta[['_id', count_cn]], on='_id', how='left')
    describe.plot_cluster_box_grouped(df, 'cluster', count_cn)
