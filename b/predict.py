import xgboost as xgb
from sklearn.metrics import auc, roc_curve
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from constants import *


def hiv_model_mask(X):
    mask = X['form-*-case-*-update-*-hiv_status->-last_known_positive'].values == 1
    mask |= X['form-*-case-*-update-*-hiv_status->-last_positive'].values == 1
    mask |= X['form-*-case-*-update-*-hiv_status->-last_tested_positive'].values == 1
    mask |= X['form-*-case-*-update-*-hiv_status->-last_unknown'].values == 1
    mask2 = X['will_have_birth_and_hiv_positive_child_0.0'].values == 1
    mask2 |= X['will_have_birth_and_hiv_positive_child_1.0'].values == 1
    sort_cns = [CASE_ID_CN, DATE_CN]
    X_subset = X.iloc[mask & mask2]
    X_subset[DATE_CN] = pd.to_datetime(X_subset[DATE_CN])
    X_subset.sort_values(by=sort_cns, inplace=True, ascending=True)
    X_subset.index = np.arange(X_subset.shape[0])
    mask3 = X[FORM_ID_CN].isin(X_subset.groupby(CASE_ID_CN).last()[FORM_ID_CN].unique()).values
    return mask & mask2 & mask3


def acfu_increment_model_mask(X, form_count=2, form_type='AN Appointment Diary', hiv_pos_only=False, sa_only=False):

    mask = np.ones(X.shape[0], dtype=bool)

    if form_type is not None:
        mask1 = X['form-*-@name'].values == form_type
        mask &= mask1

    if form_count is not None:
        mask2 = X['form_count->-current'].values.astype(float) == form_count
        mask &= mask2

    if hiv_pos_only:
        mask3 = X['form-*-case-*-update-*-hiv_status->-last_known_positive'].values == 1
        mask3 |= X['form-*-case-*-update-*-hiv_status->-last_positive'].values == 1
        mask3 |= X['form-*-case-*-update-*-hiv_status->-last_tested_positive'].values == 1
        mask3 |= X['form-*-case-*-update-*-hiv_status->-last_unknown'].values == 1
        mask &= mask3

    if sa_only:
        mask4 = X['form-*-case-*-update-*-country->-last_South Africa'].values == 1
        mask4 |= X['form-*-case-*-update-*-country->-current_South Africa'].values == 1
        mask &= mask4

    return mask


def main(X, y_cn, meta=None, plots=True, verbose=True):
    X = X.copy()
    if 'form_count->-current' in X.columns:
        X['form_count->-current'] = X['form_count->-current'].astype(float)
    y = X[y_cn].values
    del(X[y_cn])
    split_key = X[CASE_ID_CN].values
    for cn in META_CNS:
        if cn in X:
            del(X[cn])
    result = split(X, y, meta=meta, split_label=split_key, verbose=True)
    result = train_gbm(result)
    fi = pd.Series({v:i for v, i in zip(X.columns, result['gbm'].feature_importances_)}).sort_values(ascending=False)
    print('--> Top 30 features:\n{}'.format(fi.iloc[:30]))

    if plots:
        cumsum = pd.DataFrame({'score': result['test_pred'], 'target': result['y_split'][2]})
        cumsum.sort_values(by='score', ascending=False, inplace=True)
        ylabel = 'Cumulative {}'.format(y_cn)
        cumsum[ylabel] = cumsum['target'].cumsum().values
        title = 'Predictive prioritisation'
        ax = cumsum.plot(x=np.arange(cumsum.shape[0]), y=ylabel, title=title)
        ax.set(xlabel='Forms by order of decreasing score')
        plt.savefig(os.path.join(GRAPHS_FOLDER, title+'.png'), dpi=400)
        plt.close('all')

        # TODO review / test
        #select_mask = result['y'] == 1.0
        #top_var_cns = fi.index.tolist()
        #top_var_cns = [cn for cn in top_var_cns if 'nan' not in cn]
        #top_var_cns = top_var_cns[:15][::-1]
        #X_select = result['X'].iloc[select_mask][top_var_cns]
        #X_rest = result['X'].iloc[~select_mask][top_var_cns]
        #data_select = np.zeros((X_select.shape[1], X_select.shape[0]), dtype=int)    
        #data_rest = np.zeros((X_rest.shape[1], X_rest.shape[0]), dtype=int)    
        #for i, cn in enumerate(top_var_cns):
        #    data_select[i] = X_select[cn].values.astype(int)
        #    data_rest[i] = X_rest[cn].values.astype(int)
        #title = 'Some explanatory variables by target group'

        # TODO refactor / test
        #fig, ax = plt.subplots()
        #p_select = plt.plot(data_select.mean(axis=1), np.arange(len(top_var_cns)), 'o', color='#d33682')
        #p_rest = plt.plot(data_rest.mean(axis=1), np.arange(len(top_var_cns)), 'o', color='#268bd2')
        #plt.grid(True, axis='y', alpha=0.5)
        #plt.title(title)
        #plt.xlabel('incidence')
        #cleaned_cns = [cn.replace('last_country_', '') for cn in top_var_cns]
        #cleaned_cns = [cn.replace('_status_', '_') for cn in cleaned_cns]
        #cleaned_cns = [cn.replace('.0', '') for cn in cleaned_cns]
        #cleaned_cns = [cn.replace('time_since_', '') for cn in cleaned_cns]
        #wrapped_cns = ['\n'.join(textwrap.wrap(cn, 30)) for cn in cleaned_cns]
        #plt.yticks(np.arange(len(top_var_cns)), wrapped_cns)
        #plt.legend((p_select[0], p_rest[0]), ('will acfu inc.', 'rest'))
        #fig.subplots_adjust()
        #plt.tight_layout()
        #plt.savefig('{}.png'.format(title), dpi=400)
        #plt.close('all')

    return result


def split(X, y, meta=None, valid_size=0.2, test_size=0.2, split_label=None,
        random_seed=51, verbose=True):
    """
    :param split_label: split sets will be mutually exclusive on this label

    """
    assert type(split_label) in (np.ndarray, list, tuple)
    assert len(split_label) == X.shape[0]
    assert valid_size + test_size < 1.0
    kiu = np.unique(split_label)
    if random_seed is not None:
        np.random.seed(random_seed)
    np.random.shuffle(kiu)
    kiutest_size = int(test_size * kiu.shape[0])
    kiutest = kiu[:kiutest_size]
    kiuvalid_size = int(valid_size * kiu.shape[0])
    kiuvalid = kiu[kiutest_size: kiutest_size + kiuvalid_size]
    test_mask = np.in1d(split_label, kiutest)
    valid_mask = np.in1d(split_label, kiuvalid)
    train_mask = ~ (valid_mask | test_mask)
    masks_and = train_mask & valid_mask & test_mask
    masks_or = train_mask | valid_mask | test_mask
    msg = 'Masks must be disjoint'
    assert masks_and.sum() == 0 and masks_or.sum() == X.shape[0], msg
    if verbose:
        msg = '--> Sizes: train {}'.format(train_mask.sum())
        msg += ', valid: {}'.format(valid_mask.sum())
        msg += ', test: {}'.format(test_mask.sum())
        print(msg)
    X_split = X.iloc[train_mask], X.iloc[valid_mask], X.iloc[test_mask]
    y_split = y[train_mask], y[valid_mask], y[test_mask]
    if meta is not None:
        meta_split = meta.iloc[train_mask], meta.iloc[valid_mask], meta.iloc[test_mask]
    if verbose:
        msg = '---> Target outcome incidence rates:\n'
        msg += '     train: {0:.2f} ({1:.0f})\n'.format(y_split[0].sum() / y_split[0].shape[0], y_split[0].sum())
        msg += '     validation: {0:.2f} ({1:.0f})\n'.format(y_split[1].sum() / y_split[1].shape[0], y_split[1].sum())
        msg += '     test: {0:.2f} ({1:.0f})\n'.format(y_split[2].sum() / y_split[2].shape[0], y_split[2].sum())
        print(msg)
    result = {'X_split': X_split, 'y_split': y_split}
    if meta is not None:
        result.update({'meta_split': meta_split})
    return result


def train_gbm(result, plots=True, verbose=True):
    eval_set = ((result['X_split'][0], result['y_split'][0]), (result['X_split'][1], result['y_split'][1]))
    result['gbm'] = xgb.XGBClassifier(n_estimators=150, max_depth=4, nthread=-1)
    result['gbm'].fit(result['X_split'][0], result['y_split'][0], eval_set=eval_set, eval_metric='logloss',
            early_stopping_rounds=20, verbose=verbose)
    ntree_limit = result['gbm']._Booster.best_ntree_limit
    result['test_pred'] = result['gbm'].predict_proba(result['X_split'][2], ntree_limit=ntree_limit)[:, 1]
    fpr, tpr, thr = roc_curve(result['y_split'][2], result['test_pred'])
    result['auc'] = auc(fpr, tpr)
    if verbose:
        print('-> Test auc: {}'.format(result['auc']))
    if not plots:
        return result
    plt.plot(fpr, tpr, color='#268bd2', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(GRAPHS_FOLDER, 'roc.png'), dpi=400)
    plt.close('all')
    return result
