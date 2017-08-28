import json
import typing
import gc
import os
from functools import reduce, partial

import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import multiprocessing as mp

from . import helpers

NWORKERS = mp.cpu_count()
JOB_SIZE = 10000


def get_flat(jsons_path=None, form_dicts=None, vdescs=None):
    assert (jsons_path is None) != (form_dicts is None)
    assert not vdescs is None
    items = helpers.json_file_paths(jsons_path) if form_dicts is None else form_dicts
    nchunks = int(len(items) / min(len(items), JOB_SIZE))
    chunks = np.array_split(items, nchunks)
    job = job_offline if form_dicts is None else job_live

    tpaths = get_tpaths(vdescs)
    tpathss = split_tpaths(tpaths)
    job = partial(job, cns=tpaths, tpathss=tpathss)
    result = pd.DataFrame(columns=tpaths)
    with tqdm(total=len(chunks), desc='Scanning jsons', smoothing=0.1, mininterval=0.5) as pbar:
        with mp.Pool(NWORKERS) as pool:
            for df in pool.imap_unordered(job, chunks):
                result = result.append(df, ignore_index=True)
                gc.collect()
                pbar.update(1)
    return result


def job_offline(dpaths, cns, tpathss):
    ds = []
    for dpath in dpaths:
        try:
            d = json.load(open(dpath, 'r'))
        except Exception as e:
            print('-> Error while opening {}'.format(dpath))
            raise e
        ds.append(get_record(d, cns, tpathss))
    result = pd.DataFrame(ds)
    gc.collect()
    return result


def job_live(form_dicts, cns, tpathss):
    ds = []
    for d in form_dicts:
        ds.append(get_record(d, cns, tpathss))
    result = pd.DataFrame(ds)
    gc.collect()
    return result


def get_tpath(vdesc):
    path = vdesc['tree_path']
    path = path.replace('root{}'.format(helpers.TREE_PATH_EDGE), '')
    path = path.replace('root', '')
    if path != '':
        path += helpers.TREE_PATH_EDGE
    path += vdesc['name']
    return path


def get_tpaths(vdescs):
    return [get_tpath(vd) for _, vd in vdescs.iterrows()]


def split_tpaths(tpaths):
    tpathss = [tp.split(helpers.TREE_PATH_EDGE) for tp in tpaths]
    return tpathss


def get_record(droot, cns, tpathss):
    r = {}
    for cn, tp in zip(cns, tpathss):
        r[cn] = get_variable(droot, tp)
    return r


def get_variable(d, tp):
    if type(d) != dict:
        return None
    v = d.get(tp[0])
    if len(tp) == 1 or v is None:
        return v
    return get_variable(v, tp[1:])


def discard_vars(vdescs):
    mask = (vdescs['level'].values==0) & vdescs['name'].isin(['_rev','#export_tag','_attachments','auth_context','external_blobs','last_sync_token','migrating_blobs_from_couch','openrosa_headers']).values
    mask |= ((vdescs['type'].values=='list') & (~(vdescs['name'].values=='__retrieved_case_ids'))) | (vdescs['tree_path'].str.contains('\*-\[\]-\*').values)
    mask |= ((~vdescs['type'].isin(['list', 'dict']).values) & (vdescs['values'].apply(lambda x: len(x)).values==0)) | (vdescs['type'].isin(['list', 'dict']).values & (vdescs['size'].values == 0))
    mask |= vdescs['type'].values == 'dict'
    return vdescs.iloc[~mask]
