import gc

from . import current
from . import lifetime
from . import window
from . import acfu
from . import ngram
from . import infant_hiv

from constants import *
from . import helpers


def extract(dfi, meta, verbose=True):
    """
    
    Extracts all features.
    :returns: raw extraction, one-hot encoded extraction, filtered form subset
    (representing last-in-case where case at least 3 months old.

    """

    rdf = meta.copy()
    rdfd = meta.copy()
    gc.collect()

    df_lifetime, df_lifetime_dummies = lifetime.extract(dfi, verbose=verbose)
    rdf = rdf.merge(df_lifetime, on=FORM_ID_CN, how='left')
    rdfd = rdfd.merge(df_lifetime_dummies, on=FORM_ID_CN, how='left')
    gc.collect()

    df_current, df_current_dummies = current.extract(dfi, verbose=verbose)
    rdf = rdf.merge(df_current, on=FORM_ID_CN, how='left')
    rdfd = rdfd.merge(df_current_dummies, on=FORM_ID_CN, how='left')
    gc.collect()

    df_acfu, df_acfu_dummies = acfu.extract(dfi, verbose=verbose)
    rdf = rdf.merge(df_acfu, on=FORM_ID_CN, how='left')
    rdfd = rdfd.merge(df_acfu_dummies, on=FORM_ID_CN, how='left')
    gc.collect()

    df_window, df_window_dummies = window.extract(dfi, verbose=verbose)
    rdf = rdf.merge(df_window, on=FORM_ID_CN, how='left')
    rdfd = rdfd.merge(df_window_dummies, on=FORM_ID_CN, how='left')
    gc.collect()

    df_ngram = ngram.extract(dfi, verbose=verbose)
    rdf = rdf.merge(df_ngram, on=FORM_ID_CN, how='left')
    rdfd = rdfd.merge(df_ngram, on=FORM_ID_CN, how='left')
    gc.collect()

    df_infant_hiv, df_infant_hiv_dummies = infant_hiv.extract(dfi, verbose=verbose)
    rdf = rdf.merge(df_infant_hiv, on=FORM_ID_CN, how='left')
    rdfd = rdfd.merge(df_infant_hiv_dummies, on=FORM_ID_CN, how='left')
    gc.collect()

    maturity = helpers.case_maturity(dfi)
    maturity_mask = rdf[[CASE_ID_CN]].merge(maturity, on=CASE_ID_CN, how='left')['mature'].values
    rdf_mature_last = rdf.iloc[maturity_mask].groupby(CASE_ID_CN).last().reset_index()
    rdfd_mature_last = rdfd.iloc[maturity_mask].groupby(CASE_ID_CN).last().reset_index()

    exclude = [cn for cn in rdfd_mature_last.columns if cn.endswith('_None') or cn.endswith('_nan')]
    for cn in exclude:
        del(rdfd_mature_last[cn])
        del(rdfd[cn])

    return rdf, rdfd, rdf_mature_last, rdfd_mature_last
