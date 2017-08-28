from typing import List
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import a, b
import constants


def get_live_prediction(forms: List[dict]) -> List[float]:
    flat_df = a.main.live_extract_flat_forms(forms)
    dfi = b.helpers.get_input_df(constants.CURATED_SOURCE_CNS, clean=True, source_df=flat_df, verbose=True)
    meta = b.helpers.get_input_df(constants.META_CNS, clean=True, source_df=flat_df, verbose=True)
    rdf, rdfd, rdf_mature_last, rdfd_mature_last = b.curated.main.extract(dfi, meta, verbose=True)
    gbm = pickle.load(open(constants.ACFU_MODEL_PATH, 'rb'))
    fns = gbm._Booster.feature_names
    for fn in fns:
        if fn not in rdfd.columns:
            rdfd[fn] = np.nan
    mask = b.predict.acfu_increment_model_mask(rdfd)
    rdfd = rdfd.iloc[mask]
    result = rdfd[[constants.CASE_ID_CN, constants.FORM_ID_CN]]
    rdfd['form_count->-current'] = rdfd['form_count->-current'].astype(float)
    X = rdfd[fns]
    if X.shape[0] == 0:
        raise Exception('Not enough information for prediction')
    result['risk_score'] = gbm.predict_proba(X)[:, 1]
    return result


def join_by_case(df, cns, flat=None):
    assert constants.FORM_ID_CN in df.columns
    assert constants.CASE_ID_CN in df.columns
    if constants.DATE_CN not in cns:
        cns.append(constants.DATE_CN)
    if constants.CASE_ID_CN not in cns:
        cns.append(constants.CASE_ID_CN)
    if flat is None:
        flat = pd.read_csv(constants.FLAT_FORMS_PATH, usecols=cns + [constants.DATE_CN])
    else:
        assert all([cn in flat.columns for cn in cns])
    result = df.copy()
    flat.sort_values(by=[constants.CASE_ID_CN, constants.DATE_CN], inplace=True, ascending=True) 
    lambda x : x[pd.notnull(x)]
    for cn in tqdm(set(cns) - set([constants.FORM_ID_CN, constants.DATE_CN, constants.CASE_ID_CN])):
        mask = flat[cn].notnull().values
        other = flat.iloc[mask][[constants.CASE_ID_CN, cn]].groupby(constants.CASE_ID_CN).last()
        other = other.reset_index()
        result = result.merge(other, on=constants.CASE_ID_CN, how='left')
    return result


def get_unique_case_ids():
    df = pd.read_csv(constants.FLAT_FORMS_PATH, usecols=[constants.CASE_ID_CN])
    return df[constants.CASE_ID_CN].unique()


def get_by_case_ids(case_ids):
    df_it = pd.read_csv(constants.FLAT_FORMS_PATH, iterator=True, chunksize=10000, dtype=str)
    result = []
    for chunk in df_it:
        matching = chunk.iloc[chunk[constants.CASE_ID_CN].isin(case_ids).values]
        result.append(matching)
    result = pd.concat(result, ignore_index=True, copy=False)
    return result
