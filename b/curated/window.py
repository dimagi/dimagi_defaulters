import numpy as np
import pandas as pd
from tqdm import trange

from constants import *


def extract(dfi, window_size=3, verbose=True):
    for cn in dfi.columns:
        if dfi[cn].nunique(dropna=False) <= 1:
            print('---> Warning: no information in input column {}'.format(cn))
    if verbose:
        print('--> Extracting fixed-length historical window...')
    sort_cns = [CASE_ID_CN, DATE_CN]
    dfi[DATE_CN] = pd.to_datetime(dfi[DATE_CN])
    dfi.sort_values(by=sort_cns, inplace=True, ascending=True)
    dfi.index = np.arange(dfi.shape[0])
    rdf = dfi[[FORM_ID_CN]].copy()

    dates = dfi[DATE_CN].tolist()
    cases = dfi[CASE_ID_CN].values

    window_hour = np.array([[None] * dfi.shape[0]] * window_size, dtype=object)
    window_day_of_week = np.array([[None] * dfi.shape[0]] * window_size, dtype=object)

    case_offset = 0

    for i in trange(dfi.shape[0]):
        if i > 0:
            if cases[i-1] != cases[i]:
                case_offset = i
        for window_idx, j in enumerate(np.arange(i+1)[::-1][:window_size]):
            if j < case_offset:
                break
            window_hour[window_idx][i] = dates[j].hour
            window_day_of_week[window_idx][i] = dates[j].dayofweek

    cns_to_dummify = []
    for widx in range(window_size):
        cn = 'hour_{}'.format(widx if widx == 0 else -widx)
        rdf[cn] = window_hour[widx]
        cns_to_dummify.append(cn)
        cn = 'day_of_week_{}'.format(widx if widx == 0 else -widx)
        rdf[cn] = window_day_of_week[widx]
        cns_to_dummify.append(cn)

    rdf_dense = pd.get_dummies(rdf[cns_to_dummify].astype(str), prefix=cns_to_dummify)
    rdf_dense[FORM_ID_CN] = rdf[FORM_ID_CN].values

    assert rdf.shape[0] == dfi.shape[0]

    if verbose:
        print('--> Done extracting window.')

    return rdf, rdf_dense
