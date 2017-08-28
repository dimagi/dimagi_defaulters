from datetime import datetime
import collections
import textwrap
import locale

import tqdm
import numpy as np
import pandas as pd
import graphviz as gv

from . import helpers


# TODO needs fix due to refactor
#def get_xs_ys_form_counts(r):
#    dfi = r['dfi']
#
#    result = dict()
#
#    labels = ['lifetime', 'country and lifetime', 'baseline']
#    ys = []
#
#    y = []
#    for ewl in tqdm(np.arange(1,11)):
#        r = main.main(full_data=False, eng_data=True, modelling=True, no_country_data=True, sliding_window_size_eng=ewl, lifetime_history=True, random_seed=51, dfi=dfi, verbose=False, plots=False)
#        y.append(r['auc'])
#    print('AUCS with lifetime: {}'.format(y))
#    ys.append(y)
#
#    y = []
#    for ewl in tqdm(np.arange(1,11)):
#        r = main.main(full_data=False, eng_data=True, modelling=True, no_country_data=False, sliding_window_size_eng=ewl, lifetime_history=True, random_seed=51, dfi=dfi, verbose=False, plots=False)
#        y.append(r['auc'])
#    print('AUCS with country and lifetime: {}'.format(y))
#    ys.append(y)
#
#    y = []
#    for ewl in tqdm(np.arange(1,11)):
#        r = main.main(full_data=False, eng_data=True, modelling=True, no_country_data=True, sliding_window_size_eng=ewl, lifetime_history=False, random_seed=51, dfi=dfi, verbose=False, plots=False)
#        y.append(r['auc'])
#    print('AUCS baseline: {}'.format(y))
#    ys.append(y)
#
#    return [np.arange(10)] * 3, ys, labels


def get_x_y_timedelta_since_last_form(start_form_count=None, end_form_count=None, dfi=None, verbose=True):
    if dfi is None:
        input_mask = helpers.input_mask(verbose=verbose)
        cns = ['form-*-case-*-@case_id', 'form-*-case-*-@date_modified', 'form_count']
        dfi = helpers.get_input_df(cns, mask=input_mask, verbose=False)
        dfi['form-*-case-*-@date_modified'] = pd.to_datetime(dfi['form-*-case-*-@date_modified'])
        dfi.sort_values(by=['form-*-case-*-@case_id', 'form-*-case-*-@date_modified'], inplace=True, ascending=True)
    df = dfi.copy()
    offset = end_form_count - start_form_count
    df['days_since'] = df['form-*-case-*-@date_modified'].diff(periods=offset).dt.days
    mask = df['form_count'].values == end_form_count
    df = df.iloc[mask]
    vcs = df['days_since'].value_counts().sort_index()
    x = np.array(vcs.index.tolist())
    y = vcs.values
    return x, y, dfi


def get_x_y_form_name_sequences(dfi=None, verbose=True):
    if dfi is None:
        input_mask = helpers.input_mask()
        cns = ['form-*-case-*-@case_id', 'form-*-case-*-@date_modified', 'form_count', 'form-*-@name', 'will_acfu_increment']
        dfi = helpers.get_input_df(cns, mask=input_mask, verbose=verbose)
        dfi['form-*-case-*-@date_modified'] = pd.to_datetime(dfi['form-*-case-*-@date_modified'])
        dfi.sort_values(by=['form-*-case-*-@case_id', 'form-*-case-*-@date_modified'], inplace=True, ascending=True)
    df = dfi.groupby('form-*-case-*-@case_id').apply(lambda x: x['form-*-@name'].values.astype(str)).reset_index(name='sequence')
    df['sequence_str'] = df['sequence'].apply(lambda x: ' -> '.join(x)).values
    df = df.merge(dfi.groupby('form-*-case-*-@case_id')['form_count'].max().reset_index(name='ttl_form_count'), on='form-*-case-*-@case_id', how='left')
    return df, dfi


def generate_form_sequence_digraph(sequences, max_depth, format='svg'):
    tree = get_sequence_volume_tree(sequences)
    graph = gv.Digraph(format=format)
    populate_graph(0, 'All', graph, tree, max_depth, tree['count'])
    return graph


def populate_graph(depth, key, graph, tree, max_depth, ttl_pop):
    if tree['count'] < 100:
        return None, None
    nodename = '\n'.join(textwrap.wrap('{}'.format(key), 16)[:3])
    nodename += '\n{0}\n{1:,}'.format(' ' * depth, tree['count'])
    graph.node(nodename)
    if depth < max_depth:
        for k, v in tree.items():
            if isinstance(v, collections.Mapping):
                sub_nodename, count = populate_graph(depth+1, k, graph, v, max_depth, ttl_pop)
                if sub_nodename is not None:
                    graph.edge(nodename, sub_nodename, penwidth=str(30*count/ttl_pop))
    if depth > 0:
        return nodename, tree['count']


def get_sequence_volume_tree(sequences):
    tree = dict()
    for s in sequences:
        path = get_tree_path(s)
        tree = update_volume_tree(tree, path)
    return tree


def get_tree_path(sequence):
    result = {'count': 1}
    if len(sequence) > 0:
        result.update({sequence[0]: get_tree_path(sequence[1:])})
    return result


def update_volume_tree(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = update_volume_tree(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = d.get(k, 0) + u[k]
    return d
