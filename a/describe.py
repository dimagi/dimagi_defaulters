import json
import gc
import typing
from functools import reduce
import collections

from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import multiprocessing as mp

from . import helpers

NWORKERS = mp.cpu_count()
JOB_SIZE = 5000


def describe_objects(jsons_path=None, form_dicts=None):
    assert (jsons_path is None) != (form_dicts is None)
    items = helpers.json_file_paths(jsons_path) if form_dicts is None else form_dicts
    nchunks = int(len(items) / min(len(items), JOB_SIZE))
    chunks = np.array_split(items, nchunks)
    job = job_offline if form_dicts is None else job_live
    descriptions = []
    with mp.Pool(NWORKERS) as pool:
        with tqdm(total=len(chunks), desc='Scanning jsons...', smoothing=0.1, mininterval=1) as pbar:
            for description in pool.imap_unordered(job, chunks):
                descriptions.append(description)
                pbar.update(1)
    print('-> Final aggregation...')
    description = aggregate(descriptions)
    return description


def job_live(form_dicts):
    ds = []
    for d in form_dicts:
        ds += get_description(-1, None, {'root': d})
    result = aggregate([pd.DataFrame(ds)])
    gc.collect()
    return result


def job_offline(paths):
    ds = []
    for path in paths:
        try:
            d = json.load(open(path, 'r'))
        except Exception as e:
            print('-> Error while opening {}'.format(path))
            raise e
        ds += get_description(-1, None, {'root': d})
    result = aggregate([pd.DataFrame(ds)])
    gc.collect()
    return result


def aggregate(descriptions):
    description = descriptions[0]
    if len(descriptions) > 1:
        for df in descriptions[1:]:
            description = description.append(df, ignore_index=True)
    mask = description['values'].isnull()
    description.loc[mask, 'values'] = [set()] * mask.sum()
    gb = description.groupby(['name', 'level', 'tree_path', 'type'])
    sr_count = gb['count'].sum().reset_index()
    sr_size = gb['size'].sum().reset_index()['size']
    f = lambda x : reduce(set.union, x)
    sr_values = gb.aggregate({'values': f}).reset_index()['values']
    description = sr_count
    description['size'] = sr_size.values
    description['values'] = sr_values.values
    return description


def get_description(level, tpath, d):
    srs = []
    for k, v in d.items():
        if not type(v) in (list, dict):
            if pd.isnull(v):
                continue
        sr = {
            'name': k,
            'level': level,
            'tree_path': tpath,
            'type': type(v).__name__,
            'count': 1,
            'size': len(v) if type(v) in (list, dict) else 1
            }
        srs.append(sr)
        if type(v) == dict:
            if len(v) > 0:
                srs += get_description(level+1, helpers.new_tpath(tpath, k), v)
        elif type(v) == list:
            if len(v) > 0:
                for i, item in enumerate(v):
                    if type(item) == dict and len(item) > 0:
                        srs += get_description(level+1, helpers.new_tpath(tpath, k, list_parent=True), item)
        else:
            sr['values'] = {v,}
    return srs


def get_example_value(jsons_path, m2m_var):
    tpath = m2m_var['tree_path'].split(helpers.TREE_PATH_EDGE)[1:]
    if '[]' in tpath:
        raise NotImplementedError('Retrieving example values from within arrays not supported')
    print('-> Getting paths...')
    paths = helpers.json_file_paths(jsons_path)
    with tqdm(total=len(paths), desc='Scanning jsons', smoothing=0.1, mininterval=0.5) as pbar:
        for i, path in enumerate(paths):
            if i > 0:
                pbar.update(1)
            d = json.load(open(path, 'r'))
            end = False
            for i, node in enumerate(tpath):
                if node not in d and node != helpers.TREE_PATH_ARRAY:
                    break
                elif node == helpers.TREE_PATH_ARRAY:
                    if type(d) != list:
                        break
                    if len(d) == 0:
                        break
                    for item in d:
                        pass # TODO recursive traversal
                else:
                    d = d[node]
                if i == len(tpath) - 1:
                    end = True
            if end or len(tpath) == 0:
                if m2m_var['name'] in d:
                    v = d[m2m_var['name']]
                    if type(v).__name__ == m2m_var['type']:
                        return v, path
    return None, None


def find_case_array(jsons_path, m2m_var):
    tpath = m2m_var['tree_path'].split(helpers.TREE_PATH_EDGE)[1:]
    print('-> Getting paths...')
    paths = helpers.json_file_paths(jsons_path)
    with tqdm(total=len(paths), desc='Scanning jsons', smoothing=0.1, mininterval=0.5) as pbar:
        for i, path in enumerate(paths):
            if i > 0:
                pbar.update(1)
            d = json.load(open(path, 'r'))
            try:
                if type(d['form']['case'][0]) == list:
                    return d, path
            except:
                continue


def vars_and_types(d):
    result = dict(simple=[], dicts=[], lists=[])
    for k in d:
        if type(d[k]) == dict:
            result['dicts'].append(k)
        elif type(d[k]) == list:
            result['lists'].append(k)
        else:
            result['simple'].append(k)
    return result


def col_superset_to_json_superset(cns):
    r = {}
    for i, cn in enumerate(cns):
        pnodes = cn.split(helpers.TREE_PATH_EDGE)
        u = path_and_value_to_dict(pnodes)
        r = update(r, u)
    return r


def path_and_value_to_dict(path):
    if len(path) == 0:
        return ''
    return {path[0]: path_and_value_to_dict(path[1:])}


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            if isinstance(d, collections.Mapping):
                r = update(d.get(k, {}), v)
                d[k] = r
            else:
                d = v
        else:
            if isinstance(d, collections.Mapping):
                d[k] = v
            else:
                d = v
    return d
