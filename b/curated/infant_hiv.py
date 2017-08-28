from datetime import timedelta

import pandas as pd
from tqdm import trange, tqdm
import numpy as np

from constants import *
from . import helpers

MIN_OFFSET_TO_BIRTH_FORM = timedelta(days=2)
MIN_OFFSET_TO_BIRTH = timedelta(days=2)
INFANT_DOB_CN = 'form-*-case-*-update-*-infant_dob'
INFANT_ID_CN = 'form-*-case-*-update-*-infant_m2m_id'

POS_TEST_CNS = [
        "form-*-case-*-update-*-birth_pcr_result",
        "form-*-case-*-update-*-six_eight_week_pcr_result",
        "form-*-case-*-update-*-ten_week_pcr_result",
        "form-*-case-*-update-*-ten_month_infant_test_result",
        "form-*-case-*-update-*-twelve_month_infant_test_result",
        "form-*-case-*-update-*-thirteen_month_infant_test_result",
        "form-*-case-*-update-*-eighteen_twentyfour_month_infant_test_result"
        ]
NEG_TEST_CNS = [
        "form-*-case-*-update-*-ten_week_pcr_result",
        "form-*-case-*-update-*-ten_month_infant_test_result",
        "form-*-case-*-update-*-twelve_month_infant_test_result",
        "form-*-case-*-update-*-thirteen_month_infant_test_result",
        "form-*-case-*-update-*-eighteen_twentyfour_month_infant_test_result"
        ]
POS_VALUES = [
        'positive',
        'Positive'
        ]
NEG_VALUES = [
        'negative'
        ]


def extract(dfi, verbose=True):
    """
    Target is positive when all these conditions are met:
        1. an infant dob associated with this case will occur after the date of
        this form
        2. a child linked to the same case is tested positive in any of their
        PCR or HIV tests by 24 months.

    Target is negative when all these conditions are met:
        1. an infant dob associated with this case will occur after the date of
        this form
        2. a child linked to this case is tested negative at least once at or
        after the 10 week pcr result

    In all other forms the target will be null.
        
    """
    for cn in dfi.columns:
        if dfi[cn].nunique(dropna=False) <= 1:
            print('---> Warning: no information in input column {}'.format(cn))
    if verbose:
        print('-> Extracting mother to child HIV transmission flag')
    sort_cns = [CASE_ID_CN, DATE_CN]
    dfi.sort_values(by=sort_cns, inplace=True, ascending=True)
    dfi.index = np.arange(dfi.shape[0])

    rdf = dfi[[FORM_ID_CN]].copy()
    dates = pd.to_datetime(dfi[DATE_CN]).tolist()
    infant_dob = pd.to_datetime(dfi[INFANT_DOB_CN], errors='coerce').tolist()
    infant_id = dfi[INFANT_ID_CN].values
    has_infant_dob = pd.notnull(infant_dob)
    has_infant_dob_with_id = has_infant_dob & pd.notnull(infant_id)
    if verbose:
        msg = '---> {} out of {} forms have an infant dob with an m2m id'
        print(msg.format(has_infant_dob_with_id.sum(), has_infant_dob_with_id.shape[0]))
    cases = dfi[CASE_ID_CN].values

    will_birth_with_infant_id = np.array([np.nan] * rdf.shape[0], dtype=float)
    will_birth = np.array([np.nan] * rdf.shape[0], dtype=float)
    future_m2m_birth_infant_ids = [set() for i in range(rdf.shape[0])]
    future_infant_dobs = [set() for i in range(rdf.shape[0])]
    will_have_child_hiv_positive_id_match = np.array([np.nan] * rdf.shape[0], dtype=float)
    will_have_child_hiv_positive = np.array([np.nan] * rdf.shape[0], dtype=float)

    positive = np.zeros(rdf.shape[0], dtype=bool)
    for cn in POS_TEST_CNS:
        positive |= dfi[cn].isin(POS_VALUES).values
    print('forms with positive child test result: {}'.format(positive.sum()))

    negative = np.zeros(rdf.shape[0], dtype=bool)
    for cn in NEG_TEST_CNS:
        negative |= dfi[cn].isin(NEG_VALUES).values
    negative[positive] = False
    print('forms with eligible negative child test result: {}'.format(negative.sum()))

    assert not any(positive & negative)

    print('--> Extracting future m2m infant ids and dobs')
    for i in trange(dfi.shape[0] - 1):
        for j in np.arange(i+1, dfi.shape[0]):
            if cases[j] != cases[i]:
                break
            if not has_infant_dob[j]:
                continue
            if dates[j] <= dates[i] + MIN_OFFSET_TO_BIRTH_FORM:
                continue
            if (infant_dob[j] - dates[i]) < MIN_OFFSET_TO_BIRTH:
                continue
            will_birth[i] = 1.0
            future_infant_dobs[i].add(infant_dob[j])
            if not has_infant_dob_with_id[j]:
                continue
            future_m2m_birth_infant_ids[i].add(infant_id[j])
            will_birth_with_infant_id[i] = 1.0

    print('--> Extracting will-transmit-hiv target')
    for i in trange(dfi.shape[0] - 1):
        for j in np.arange(i+1, dfi.shape[0]):
            if cases[j] != cases[i]:
                break
            if not will_birth[i]:
                continue
            if not any([dob <= dates[j] for dob in future_infant_dobs[i]]):
                continue
            if pd.isnull(will_have_child_hiv_positive[i]) or will_have_child_hiv_positive[i] == 0:
                if positive[j]:
                    will_have_child_hiv_positive[i] = 1.0
                elif negative[j]:
                    will_have_child_hiv_positive[i] = 0.0
            if not will_birth_with_infant_id[i]:
                continue
            if infant_id[j] not in future_m2m_birth_infant_ids[i]:
                continue
            if pd.isnull(will_have_child_hiv_positive_id_match[i]) or will_have_child_hiv_positive_id_match[i] == 0:
                if positive[j]:
                    will_have_child_hiv_positive_id_match[i] = 1.0
                elif negative[j]:
                    will_have_child_hiv_positive_id_match[i] = 0.0

    rdf['will_transmit_hiv_infant_id_matched'] = will_have_child_hiv_positive_id_match
    rdf['will_have_birth_and_hiv_positive_child'] = will_have_child_hiv_positive
    rdf['will_birth_with_infant_id'] = will_birth_with_infant_id
    rdf['will_birth'] = will_birth
    rdf['current_has_positive_child_hiv_result'] = positive
    rdf['current_has_negative_child_hiv_result'] = negative

    assert rdf.shape[0] == dfi.shape[0]

    if verbose:
        print('--> Finished extracting child HIV features.')

    dum_cns = list(set(rdf.columns.tolist()) - set([FORM_ID_CN]))
    rdf_dense = pd.get_dummies(rdf[dum_cns].astype(str), prefix=dum_cns)
    rdf_dense[FORM_ID_CN] = rdf[FORM_ID_CN].values

    return rdf, rdf_dense
