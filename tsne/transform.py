import numpy as np
import pandas as pd
from bhtsne import tsne
from sklearn.cluster import DBSCAN


def dimred_and_cluster(X: pd.DataFrame, ndims: int = 2, perplexity: float = 10) -> np.ndarray:
    """ 
    Returns an `ndims`-dimension representation of `X` that preserves high-dimensional neighbourhoods, annotated with (HDBSCAN) cluster labels.

    """
    X = X.astype('float64')
    X_tsne = tsne(X, perplexity=perplexity, dimensions=ndims, rand_seed=200)
    result = pd.DataFrame(X_tsne, columns=['tsne_{}'.format(i) for i in range(ndims)])
    result['cluster'] = cluster(result)
    return result


def cluster(X: pd.DataFrame) -> np.ndarray:
    return DBSCAN(min_samples=165, eps=2.6).fit(X).labels_
