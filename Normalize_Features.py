def featureNormalize(X):
    import numpy as np
    np.asarray(X)
    mu=np.ndarray.mean(X,axis=0)
    X_norm=X-mu

    sigma=np.ndarray.std(X_norm,axis=0)
    X_norm=X_norm/sigma
    print('the mean is',mu)
    print('and sigma is',sigma)
    
    return X_norm
