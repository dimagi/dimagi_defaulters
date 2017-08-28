import gc
import functools
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm
import numpy as np

from constants import *
from . import helpers

DFI = None


def extract(dfi, verbose=True):
    """
    .. note:: `last` values in this function reflect the last-but-current-value

    """
    for cn in dfi.columns:
        if dfi[cn].nunique(dropna=False) <= 1:
            print('---> Warning: No information in input column {}'.format(cn))
    if verbose:
        print('--> Extracting lifetime features...')
    sort_cns = [CASE_ID_CN, DATE_CN]
    dfi[DATE_CN] = pd.to_datetime(dfi[DATE_CN])
    dfi.sort_values(by=sort_cns, inplace=True, ascending=True)
    dfi.index = np.arange(dfi.shape[0])

    dense_continuous_cns_passthrough = []
    cns_to_dummify = []
    for cn in DOB_INPUT_CNS:
        dfi[cn] = pd.to_datetime(dfi[cn], errors='coerce', infer_datetime_format=True)
    cn_maps = {k: dict() for k in ('last_value', 'days_since', 'time_since', 'last_dob_age', 'buckets')}
    for icn in LIFETIME_INPUT_CNS:
        if icn in DOB_INPUT_CNS:
            ocn = icn + '->-last_dob_age_years'
            cn_maps['last_dob_age'][icn] = ocn
            cns_to_dummify.append(ocn)
        else:
            ocn = icn + '->-last'
            cn_maps['last_value'][icn] = ocn
            if icn in DENSE_CONTINUOUS_CNS:
                assert dfi[icn].notnull().all()
                dense_continuous_cns_passthrough.append(ocn)
            elif icn in OTHER_BUCKET_CNS:
                ocn = ocn + '->-bucketised'
                cn_maps['buckets'][icn] = ocn
                cns_to_dummify.append(ocn)
            else:
                assert dfi[icn].nunique() < 35, 'Cardinality too high to dummify: {}'.format(icn)
                cns_to_dummify.append(ocn)
        ocn = icn + '->-days_since_last'
        cn_maps['days_since'][icn] = ocn
        ocn = icn + '->-time_since_last'
        cn_maps['time_since'][icn] = ocn
        cns_to_dummify.append(ocn)

    global DFI
    DFI = dfi.copy()
    job = functools.partial(worker, cn_maps=cn_maps)
    nworkers = mp.cpu_count() - 1
    case_subsets = np.array_split(dfi[CASE_ID_CN].unique(), nworkers * 5)
    rdfs = []
    with tqdm(total=len(case_subsets), desc='Traversing rows...', smoothing=0.05, mininterval=1) as pbar:
        with mp.Pool(nworkers) as pool:
            for srdf in pool.imap_unordered(job, case_subsets):
                rdfs.append(srdf)
                pbar.update(1)
    rdf = pd.concat(rdfs, ignore_index=True, axis=0)
    gc.collect()

    if verbose:
        print('--> Converting days-since columns to buckets...')
    for icn in LIFETIME_INPUT_CNS:
        rdf[cn_maps['time_since'][icn]] = helpers.days_to_buckets(rdf[cn_maps['days_since'][icn]].values)

    if verbose:
        print('--> Converting other high cardinality columns to buckets...')
    for icn in OTHER_BUCKET_CNS:
        buckets = None
        for k, v in BUCKETS.items():
            if icn.startswith(k):
                buckets = v
        if buckets is None:
            msg = '--> Column {} marked for discretisation but no buckets defined'
            raise Exception(msg.format(icn))
        rdf[cn_maps['buckets'][icn]] = helpers.values_to_buckets(dfi[icn].values.astype(float), buckets=buckets)

    rdf_dense = pd.get_dummies(rdf[cns_to_dummify].astype(str), prefix=cns_to_dummify)
    for cn in dense_continuous_cns_passthrough:
        rdf_dense[cn] = rdf[cn].values
    rdf_dense[FORM_ID_CN] = rdf[FORM_ID_CN].values

    assert dfi.shape[0] == rdf_dense.shape[0]

    if verbose:
        print('---> Finished extracting lifetime features.')
    return rdf, rdf_dense


def worker(case_subset, cn_maps):
    sdfi = DFI.iloc[DFI[CASE_ID_CN].isin(case_subset).values]
    dates = sdfi[DATE_CN].tolist()
    cases = sdfi[CASE_ID_CN].values

    last_nonnull_value = {cn: None for cn in cn_maps['last_value'].keys()}
    last_nonnull_value_date = {cn: None for cn in cn_maps['days_since'].keys()}

    case = None
    case_offset = 0

    rdf = sdfi[[FORM_ID_CN]].copy()
    for k in cn_maps:
        for rcn in cn_maps[k].values():
            rdf[rcn] = None

    for i in range(sdfi.shape[0]):
        if case != cases[i]:
            case = cases[i]
            case_offset = i
            for icn in LIFETIME_INPUT_CNS:
                last_nonnull_value[icn] = None
                last_nonnull_value_date[icn] = None

        for icn in LIFETIME_INPUT_CNS:

            ld = last_nonnull_value_date[icn]
            if ld is not None:
                cn_loc = rdf.columns.get_loc(cn_maps['days_since'][icn])
                rdf.iloc[i, cn_loc] = (dates[i] - ld).days

            lv = last_nonnull_value[icn]
            if lv is not None:
                if icn in DOB_INPUT_CNS:
                    cn_loc = rdf.columns.get_loc(cn_maps['last_dob_age'][icn])
                    rdf.iloc[i, cn_loc] = helpers.get_age_in_years(lv, ld)
                else:
                    cn_loc = rdf.columns.get_loc(cn_maps['last_value'][icn])
                    rdf.iloc[i, cn_loc] = lv

            tv = sdfi.iloc[i, sdfi.columns.get_loc(icn)]
            if pd.notnull(tv):
                last_nonnull_value[icn] = tv
                last_nonnull_value_date[icn] = dates[i]
    gc.collect()
    return rdf
