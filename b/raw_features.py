"""raw_features.py

This module provides functions that extract any number of raw features from the
flat view, including with a look-back of any size.

"""
# NOTE currently lightly broken (syntax errors after refactor)
# NOTE not used in final deliverables, we have given preference to curated features

from constants import *

FORM_UPDATE_MIN_CARD = 2
FORM_UPDATE_MAX_CARD = 3


def extract_raw_features(result, form_update_dense_continuous_vars, sliding_window_size_rest):
    input_mask = helpers.input_mask()
    case_ids = helpers.get_input_df([CASE_ID_CN], path=CONFIG['input_path'], mask=input_mask, dtype=str)[CASE_ID_CN].values
    form_update_vars = json.load(open(CONFIG['column_selection_path'], 'r'))['form_update']
    if form_update_dense_continuous_vars is None:
        form_update_dense_continuous_vars = get_dense_continuous_vars(form_update_vars, input_mask)
    result['form_update_dense_continuous_vars'] = form_update_dense_continuous_vars
    df_cont_dense = helpers.get_input_df(form_update_dense_continuous_vars, path=CONFIG['input_path'], mask=input_mask, dtype=None)
    if df_cont_dense.shape[0] > 0:
        df_cont_dense[CASE_ID_CN] = case_ids
        df_cont_dense = extract_window(df_cont_dense, sliding_window_size_rest, verbose=verbose)

    if sparse_or_categorical_vars is None:
        sparse_or_categorical_vars = list(set(form_update_vars) - set(form_update_dense_continuous_vars))
    result['sparse_or_categorical_vars'] = sparse_or_categorical_vars 
    if form_update_dummifiable_vars is None:
        form_update_dummifiable_vars = get_dummifiable_vars(sparse_or_categorical_vars, input_mask)
    result['form_update_dummifiable_vars'] = form_update_dummifiable_vars
    df_sparse_or_cat = helpers.get_input_df(form_update_dummifiable_vars, path=CONFIG['input_path'], mask=input_mask)
    if random_seed is not None:
        np.random.seed(random_seed)
    df_sparse_or_cat = df_sparse_or_cat.iloc[:, np.random.choice(len(df_sparse_or_cat.columns), 5)]# TODO increase sample
    df_sparse_or_cat[CASE_ID_CN] = case_ids
    df_sparse_or_cat = extract_window(df_sparse_or_cat, sliding_window_size_rest, verbose=verbose)
    df_sparse_or_cat_dum = pd.get_dummies(df_sparse_or_cat)
    return result


def extract_window(df, sliding_window_size_rest, verbose=True):
    if verbose:
        print('-> Extracting sliding window for {} columns.'.format(len(df.columns)))
    result = pd.DataFrame()
    cases = df[CASE_ID_CN].values
    last_case = None
    for cn in tqdm(df.columns):
        if cn == CASE_ID_CN:
            continue
        sr = np.array([[None] * df.shape[0]] * sliding_window_size_rest, dtype=object)
        for i in trange(df.shape[0]):
            if last_case != cases[i]:
                case_offset = i
            for j in range(sliding_window_size_rest):
                if i - j == case_offset:
                    sr[0][i] = df[cn].iloc[i]
                elif i - j >= case_offset:
                    sr[j][i] = sr[0][i-j]
            last_case = cases[i]
        for i in range(sliding_window_size_rest):
            if pd.isnull(sr[i]).all():
                continue
            result['{}_window_{}'.format(cn, i if i == 0 else -i)] = sr[i]
    return result


def get_dummifiable_vars(cns, input_mask):
    cnss = np.array_split(cns, 10)
    result = []
    for cns in tqdm(cnss):
        df = helpers.get_input_df(cns, path=CONFIG['input_path'], mask=input_mask, dtype=str)
        for cn in cns:
            nu = df[cn].nunique()
            if pd.isnull(df[cn]).any():
                nu += 1
            if nu >= FORM_UPDATE_MIN_CARD and nu <= FORM_UPDATE_MAX_CARD:
                result.append(cn)
    return result


def get_dense_continuous_vars(cns, input_mask):
    cnss = np.array_split(cns, 10)
    result = []
    for cns in tqdm(cnss):
        df = helpers.get_input_df(cns, path=CONFIG['input_path'], mask=input_mask, dtype=str)
        for cn in cns:
            try:
                _ = df[cn].astype(float)
                if pd.notnull(df[cn]).all():
                    result.append(cn)
            except ValueError as e:
                pass
    return result
