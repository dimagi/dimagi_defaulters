import subprocess
import collections

import pandas as pd
import numpy as np

from constants import *


def get_input_df(cns, source_df=None, dtype=None, verbose=False):
    if verbose:
        print('-> Reading data from {}'.format(FLAT_FORMS_PATH))
    kwargs = dict(dtype=dtype) 
    cns = list(set(cns + [CASE_ID_CN, FORM_ID_CN]))
    if source_df is not None:
        missing_cns = list(set(cns) - set(source_df.columns))
        avail_cns = list(set(cns).intersection(set(source_df.columns)))
        if len(missing_cns) > 0:
            print('---> Warning: columns missing from input:\n{}'.format(missing_cns))
            for cn in missing_cns:
                source_df[cn] = np.nan
        df = source_df
    else:
        csv_cns = list(set(cns) - set(['form_count']))
        df = pd.read_csv(FLAT_FORMS_PATH, usecols=csv_cns, dtype=dtype)
    if 'form_count' in cns:
        df = add_form_count(df, verbose=verbose)
    return df


def clean(df, verbose=False):
    assert FORM_ID_CN in df.columns
    clean_cns = [ACFU_ROUND_CN, CASE_ID_CN, FORM_TYPE_CN, COUNTRY_CN, FORM_ID_CN]
    df.index = np.arange(df.shape[0])
    if all([cn in df.columns for cn in clean_cns]):
        clean_df = df[clean_cns].copy()
    else:
        clean_df = pd.read_csv(FLAT_FORMS_PATH, usecols=clean_cns, dtype=str)
        assert clean_df[FORM_ID_CN].isin(df[FORM_ID_CN].values).all()
        clean_df = df[[FORM_ID_CN]].merge(clean_df, on=FORM_ID_CN, how='left')
    df = df.iloc[input_mask(df=clean_df, verbose=verbose)]
    return df


def add_form_count(df, verbose=False):
    sort_cns = [CASE_ID_CN, DATE_CN]
    df.index = np.arange(df.shape[0])
    if all([cn in df.columns for cn in sort_cns]):
        df.sort_values(by=sort_cns, inplace=True, ascending=True)
        df.index = np.arange(df.shape[0])
    else:
        sort_df = pd.read_csv(FLAT_FORMS_PATH, usecols=sort_cns, dtype=str)
        sort_df.index = np.arange(sort_df.shape[0])
        sort_df.sort_values(by=sort_cns, inplace=True, ascending=True)
        df = df.iloc[sort_df.index.tolist()]
    df['form_count'] = np.ones(df.shape[0], dtype=int)
    df['form_count'] = df.groupby(CASE_ID_CN)['form_count'].cumsum().values
    return df


def skiprows_idx_for_selection(path, selection, verbose=False):
    """ `selection` should be a numpy array of row numbers """
    assert type(selection) == np.ndarray
    assert selection.dtype == int
    idx = np.arange(get_nr_rows(path, verbose=verbose))
    return idx[~np.in1d(idx, selection)] + 1


def get_nr_rows(path, verbose=False):
    """ Retrieves the number of data rows in the corresponding data file
    .. note:: deducts 1 from number of lines in file to account for header

    """
    nlines = int(subprocess.check_output(['wc', '-l', path]).split()[0])
    r = max(nlines - 1, 0)
    return r


def input_mask(df=None, verbose=False):
    cns = [ACFU_ROUND_CN, CASE_ID_CN, FORM_TYPE_CN, COUNTRY_CN]
    if df is None:
        df = get_input_df(cns, clean=False, dtype=str, verbose=verbose)
    else:
        df = df[cns].copy()
    mask = ~ df.isnull().all(axis=1).values
    mask &= df[CASE_ID_CN].str.len().values == 36
    mask &= ~ df[CASE_ID_CN].isnull().values
    mask &= ~ df[FORM_TYPE_CN].isnull().values
    exclude_case_mask = df[COUNTRY_CN].values.astype(str) == '(Demo) Australia'
    exclude_cases = df.iloc[exclude_case_mask][CASE_ID_CN].unique()
    mask &= ~ df[CASE_ID_CN].isin(exclude_cases).values
    return mask
