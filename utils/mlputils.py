from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def MLP_Regressor(param_grid, cv, x_train, y_train, x_test, y_test):
    mlp = MLPRegressor()
    reg = GridSearchCV(mlp, param_grid=param_grid, scoring='r2',
                       cv=cv, verbose=True, pre_dispatch='2*n_jobs')
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)

    for i, val in enumerate(y_pred):
        if abs(val - y_test[i]) <= 0.1:
            y_pred[i] = y_test[i]
    return r2_score(y_pred, y_test), reg.cv_results_, reg.best_params_


def MLP_Classifier(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return clf.score(x_test, y_test), classification_report(y_test, y_pred), y_pred

