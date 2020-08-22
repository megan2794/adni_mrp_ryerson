from sklearn import mixture
import numpy as np

def processing_gmm(df, method='aic'):

    num_cols = len(df.columns)
    x = df.drop(['DX'], axis=1)
    n_components = np.arange(1, num_cols)
    models = [mixture.GaussianMixture(n, covariance_type='full', reg_covar=1e-5, max_iter=100).fit(x)
              for n in n_components]

    if method == 'aic':
        aic = [m.aic(x) for m in models]
        return aic.index(min(aic))
    else:
        bic = [m.bic(x) for m in models]
        return bic.index(min(bic))
