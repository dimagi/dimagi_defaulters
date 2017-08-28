from datetime import datetime
import functools
import multiprocessing as mp
import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import *
from . import helpers

DFI = None


def extract(dfi, verbose=True):
    for cn in dfi.columns:
        if dfi[cn].nunique(dropna=False) <= 1:
            print('---> Warning: No information in input column {}'.format(cn))
    if verbose:
        print('-> Extracting features from current form...')
    sort_cns = [CASE_ID_CN, DATE_CN]
    dfi[DATE_CN] = pd.to_datetime(dfi[DATE_CN])
    dfi.sort_values(by=sort_cns, inplace=True, ascending=True)
    dfi.index = np.arange(dfi.shape[0])

    cn_maps = {k: dict() for k in ('dob_age', 'event_date_timedelta_days', 'event_date_timedelta_buckets', 'current_value', 'buckets')}
    dense_continuous_cns_passthrough = []
    cns_to_dummify = []
    for icn in CURRENT_FORM_INPUT_CNS:
        if icn in DOB_INPUT_CNS:
            ocn = icn + '->-current_dob_age_years'
            cn_maps['dob_age'][icn] = ocn
            cns_to_dummify.append(ocn)
        elif icn in EVENT_DATE_INPUT_CNS:
            ocn = icn + '->-current_delta_days'
            cn_maps['event_date_timedelta_days'][icn] = ocn
            ocn = icn + '->-current_delta_time'
            cn_maps['event_date_timedelta_buckets'][icn] = ocn
            cns_to_dummify.append(ocn)
        else:
            ocn = icn + '->-current'
            cn_maps['current_value'][icn] = ocn
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

    for icn in DOB_INPUT_CNS:
        dfi[icn] = pd.to_datetime(dfi[icn], errors='coerce', infer_datetime_format=True)

    for icn in EVENT_DATE_INPUT_CNS:
        dfi[icn] = pd.to_datetime(dfi[icn], errors='coerce', infer_datetime_format=True)
        mask = (dfi[icn] < datetime(2000, 1, 1)).values
        dfi.iloc[mask, dfi.columns.get_loc(icn)] = None

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
    rdf = pd.concat(rdfs)
    gc.collect()

    if verbose:
        print('--> Converting days-since columns to buckets...')
    for icn in EVENT_DATE_INPUT_CNS:
        rdf[cn_maps['event_date_timedelta_buckets'][icn]] = helpers.days_to_buckets(rdf[cn_maps['event_date_timedelta_days'][icn]].values)

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

    assert rdf.shape[0] == dfi.shape[0]

    if verbose:
        print('---> Finished extracting features from current form.')

    return rdf, rdf_dense


def worker(case_subset, cn_maps):
    sdfi = DFI.iloc[DFI[CASE_ID_CN].isin(case_subset).values]
    rdf = sdfi[[FORM_ID_CN]].copy()
    for k in cn_maps:
        for rcn in cn_maps[k].values():
            rdf[rcn] = None

    dates = sdfi[DATE_CN].tolist()

    for i in range(sdfi.shape[0]):
        for icn in CURRENT_FORM_INPUT_CNS:
            icn_loc = sdfi.columns.get_loc(icn)
            if icn in DOB_INPUT_CNS:
                dob_date = sdfi.iloc[i, icn_loc]
                rcn_loc = rdf.columns.get_loc(cn_maps['dob_age'][icn])
                rdf.iloc[i, rcn_loc] = helpers.get_age_in_years(dob_date, dates[i])
            elif icn in EVENT_DATE_INPUT_CNS:
                event_date = sdfi.iloc[i, icn_loc]
                rcn_loc = rdf.columns.get_loc(cn_maps['event_date_timedelta_days'][icn])
                rdf.iloc[i, rcn_loc] = helpers.get_timedelta_days(event_date, dates[i])
            else:
                rcn_loc = rdf.columns.get_loc(cn_maps['current_value'][icn])
                rdf.iloc[i, rcn_loc] = sdfi.iloc[i, icn_loc]
    gc.collect()
    return rdf
