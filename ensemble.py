from sklearn.model_selection import train_test_split
import numpy as np

def ensemble_features(X_train, y_train, X_test, ratio, proba=0, random_state=None, *models):

    '''
    Function that create new features based on an ensemble of models
    Parameters:
    -----------
    Inputs:
        X_train (dataframe, array): train feature matrix
        y_train (series, array): train label vector
        X_test (dataframe, array): test feature matrix
        ratio (float): proportion of data rows to maintain on the new ensemble X,y
        random_state (int): random seed
        models (model): models to become new ensemble features
    Returns:
        X_train_ensemble (array): new X_train feature matrix
        y_train_ensemble (array): new y_train label vector
        X_test_ensemble (array): new X_test feature matrix
    '''

    n_models = len(models)
    if ratio == 0:
        X_train_, X_test_, y_train_, y_train_ensemble = X_train, X_train, y_train, y_train
    else:
        X_train_, X_test_, y_train_, y_train_ensemble = train_test_split(X_train,
                                                                        y_train,
                                                                        test_size=ratio,
                                                                        stratify=y_train,
                                                                        random_state=random_state)

    X_train_ensemble = np.zeros((X_test_.shape[0],n_models))
    X_test_ensemble = np.zeros((X_test.shape[0], n_models))

    for i, model in enumerate(models):

        model.fit(X_train_, y_train_)
        if proba == 0:   
            X_train_ensemble[:,i] = model.predict(X_test_)
            X_test_ensemble[:,i] = model.predict(X_test)
        elif proba == 1:
            X_train_ensemble[:,i] = model.predict_proba(X_test_)[:,1]
            X_test_ensemble[:,i] = model.predict_proba(X_test)[:,1]      

    return X_train_ensemble, y_train_ensemble, X_test_ensemble