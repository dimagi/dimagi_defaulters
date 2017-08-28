from datetime import timedelta

import pandas as pd
from tqdm import trange, tqdm
import numpy as np

from constants import *
from . import helpers


def extract(dfi, window_ndays=90, verbose=True):
    """ Extracts current ACFU round and ACFU target in `window_ndays` look-ahead """
    for cn in dfi.columns:
        if dfi[cn].nunique(dropna=False) <= 1:
            print('---> Warning: No information in input column {}'.format(cn))
    if verbose:
        print('-> Extracting acfu will-increment-target ({} day window)...'.format(window_ndays))
    sort_cns = [CASE_ID_CN, DATE_CN]
    dfi.sort_values(by=sort_cns, inplace=True, ascending=True)
    dfi.index = np.arange(dfi.shape[0])

    rdf = dfi[[FORM_ID_CN]].copy()
    dates = pd.to_datetime(dfi[DATE_CN]).tolist()
    cases = dfi[CASE_ID_CN].values
    acfu_rounds = dfi['form-*-case-*-update-*-acfu_round'].values.astype(float)

    rdf['current_is_acfu_increment'] = np.zeros(dfi.shape[0])
    rdf['days_since_last_is_acfu_increment'] = np.nan
    last_incr_date = None
    if verbose:
        print('--> First getting current acfu increment flags...')
    for i in trange(dfi.shape[0]):
        if i == 0:
            continue
        if cases[i] != cases[i-1]:
            last_incr_date = None
            continue
        if last_incr_date is not None:
            delta = dates[i] - last_incr_date
            rdf.iloc[i, rdf.columns.get_loc('days_since_last_is_acfu_increment')] = delta.days
        if pd.isnull(acfu_rounds[i]) or acfu_rounds[i] == 0:
            continue
        if pd.notnull(acfu_rounds[i-1]) and acfu_rounds[i-1] == acfu_rounds[i]:
            continue
        rdf.iloc[i, rdf.columns.get_loc('current_is_acfu_increment')] = 1
        last_incr_date = dates[i]

    will_incr_cn = 'will_acfu_incr_within_{}_days'.format(window_ndays)
    rdf[will_incr_cn] = np.zeros(dfi.shape[0], dtype=float)
    acfu_incr = rdf['current_is_acfu_increment'].values
    if verbose:
        print('--> Getting lookahead acfu increment flag...')
    for i in trange(dfi.shape[0] - 1):
        for j in np.arange(i+1, dfi.shape[0]):
            if cases[j] != cases[i]:
                break
            elif dates[j] == dates[i]:
                continue
            elif dates[j] > (dates[i] + timedelta(days=window_ndays)):
                break
            elif acfu_incr[j] == 1.0:
                rdf.iloc[i, rdf.columns.get_loc(will_incr_cn)] = 1.0
                break
            else:
                continue

    if verbose:
        print('--> Converting days_since_last_acfu_incr to buckets...')
    rdf['time_since_last_is_acfu_increment'] = helpers.days_to_buckets(rdf['days_since_last_is_acfu_increment'].values)

    dum_cns = ['time_since_last_is_acfu_increment']
    rdf_dummies = pd.get_dummies(rdf[dum_cns].astype(str), prefix=dum_cns)
    rdf_dummies['current_is_acfu_increment'] = rdf['current_is_acfu_increment'].values
    rdf_dummies[will_incr_cn] = rdf[will_incr_cn].values
    rdf_dummies[FORM_ID_CN] = rdf[FORM_ID_CN].values

    assert rdf.shape[0] == dfi.shape[0]

    if verbose:
        print('--> Finished extracting acfu features.')

    return rdf, rdf_dummies
