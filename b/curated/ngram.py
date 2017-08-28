import gc
import functools
import multiprocessing as mp
import typing

import pandas as pd
from tqdm import tqdm, trange
import numpy as np

from constants import NGRAM_CNS, CASE_ID_CN, DATE_CN, META_DATE_CNS, FORM_ID_CN
from . import helpers

DFI = None


def extract(dfi, ns=[2], verbose=True):
    """

    """
    for cn in dfi.columns:
        if dfi[cn].nunique(dropna=False) > 1:
            print('---> Warning: no information in input column {}'.format(cn))
    if verbose:
        print('--> Extracting ngram features...')
    sort_cns = [CASE_ID_CN, DATE_CN]
    dfi[DATE_CN] = pd.to_datetime(dfi[DATE_CN])
    dfi.sort_values(by=sort_cns, inplace=True, ascending=True)
    dfi.index = np.arange(dfi.shape[0])
    ngram_cns = NGRAM_CNS.copy()

    for icn in set(META_DATE_CNS).intersection(set(ngram_cns)):
        ngram_cns.remove(icn)
        dfi[icn] = pd.to_datetime(dfi[icn], errors='coerce', infer_datetime_format=True)
        nicn = icn + '->-day_of_week'
        ngram_cns.append(nicn)
        dfi[nicn] = dfi[icn].dt.dayofweek.values
        nicn = icn + '->-hour_of_day'
        ngram_cns.append(nicn)
        dfi[nicn] = dfi[icn].dt.hour.values

    global DFI
    DFI = dfi

    job = functools.partial(extract_column, ns=ns)
    sources = [(dfi[cn].values, cn) for cn in ngram_cns]
    results = []
    for result in map(job, sources):
        results.append(result)
    rdf = DFI[[FORM_ID_CN]].copy()
    for sr in results:
        for cn, v in sr.items():
            rdf[cn] = v 

    assert dfi.shape[0] == rdf.shape[0]

    if verbose:
        print('---> Finished extracting ngram features.')
    return rdf


def get_ngrams(
        x: np.ndarray, n: int, pad_left: bool = False, pad_right: bool = False,
        pad:typing.Any = None, as_str: bool = True
        ) -> np.ndarray:
    """ Returns all sequences of length `n` in `x`

    :param pad_left: prepend `x` with an array of `pad` of length `n-1`
    :param pad_right: same, but appended to `x`
    
    """
    if pad_left:
        x = np.concatenate(([pad] * (n - 1), x))
    if pad_right:
        x = np.concatenate((x, [pad] * (n - 1)))
    result = list(zip(*[x[i:] for i in range(n)]))
    if as_str:
        old_result = result.copy()
        for i, item in enumerate(old_result):
            result[i] = '-+-'.join(tuple([str(sitem) for sitem in item]))
    return result


def extract_column(source, ns, verbose=True):
    source, source_cn = source
    if verbose:
        print('---> Extracting ngrams for column {}'.format(source_cn))
    assert type(source) == np.ndarray
    assert len(source.shape) == 1
    cases = DFI[CASE_ID_CN].values
    ngram_result = dict()
    rsize = cases.shape[0]

    for sn in ns:
        ocn_prefix = source_cn + '->-{0:.0f}gram'.format(sn)
        case = None
        for i in trange(rsize):

            if case != cases[i]:
                case = cases[i]
                case_offset = i

            ngrams = get_ngrams(source[case_offset: i+2], sn, pad_left=True, pad_right=True, pad=None, as_str=True)
            for ngram, count in zip(*np.unique(ngrams, return_counts=True)):
                assert type(count) == np.int64
                ocn = ocn_prefix + '->-{}'.format(ngram)
                if ocn not in ngram_result:
                    ngram_result[ocn] = np.zeros(rsize, dtype=float)
                ngram_result[ocn][i] = count
    return ngram_result
