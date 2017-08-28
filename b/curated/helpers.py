from dateutil import relativedelta

import numpy as np
import pandas as pd
from constants import *


def days_to_buckets(v, buckets=[0, 7, 14, 30, 60, 120, 300, 1000]):
    result = np.array([None] * v.shape[0], dtype=object)
    if pd.isnull(v).sum() > 0:
        v[pd.isnull(v)] = np.nan
    buckets_idxs = {}
    buckets_idxs['zero_days'] = v == 0
    for i in range(len(buckets)-1):
        nv = '{}_to_{}_days'.format(buckets[i]+1, buckets[i+1])
        buckets_idxs[nv] = (v > buckets[i]) & (v <= buckets[i+1])
    buckets_idxs['more_than_{}_days'.format(buckets[-1]+1)] = v > buckets[-1]
    for b, idx in buckets_idxs.items():
        result[idx] = b
    return result.astype(str)


def values_to_buckets(v, buckets):
    result = np.array([None] * v.shape[0], dtype=object)
    if pd.isnull(v).sum() > 0:
        v[pd.isnull(v)] = np.nan
    buckets_idxs = {}
    buckets_idxs['zero'] = v == 0
    for i in range(len(buckets)-1):
        nv = '{}_to_{}'.format(buckets[i]+1, buckets[i+1])
        buckets_idxs[nv] = (v > buckets[i]) & (v <= buckets[i+1])
    buckets_idxs['more_than_{}'.format(buckets[-1]+1)] = v > buckets[-1]
    for b, idx in buckets_idxs.items():
        result[idx] = b
    return result.astype(str)


def get_age_in_years(dob, dnow):
    if pd.isnull(dob):
        return None
    result = relativedelta.relativedelta(dnow, dob).years
    return result


def get_timedelta_days(then, now):
    if pd.isnull(then):
        return None
    result = (now - then).days
    return result


def case_maturity(dfi):
    dfi = dfi[[CASE_ID_CN, DATE_CN]].copy()
    sort_cns = [CASE_ID_CN, DATE_CN]
    dfi[DATE_CN] = pd.to_datetime(dfi[DATE_CN])
    dfi.sort_values(by=sort_cns, inplace=True, ascending=True)
    dfi.index = np.arange(dfi.shape[0])

    result = pd.DataFrame({CASE_ID_CN: dfi[CASE_ID_CN].unique()})

    lasts = dfi.groupby(CASE_ID_CN)[DATE_CN].max().reset_index(name='last_date')
    result = result.merge(lasts, on=CASE_ID_CN, how='left')
    firsts = dfi.groupby(CASE_ID_CN)[DATE_CN].min().reset_index(name='first_date')
    result = result.merge(firsts, on=CASE_ID_CN, how='left')
    maturity = (result['last_date'] - result['first_date']).dt.days
    result['mature'] = (maturity >= 90) & (maturity <= (365 * 2))
    return result[[CASE_ID_CN, 'mature']]
