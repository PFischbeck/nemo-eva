import numpy

from sklearn.model_selection import \
    cross_val_score, train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

class Classification:
    def __init__(self, X, Y, model, cv_grid=None):
        assert numpy.isfinite(X).all().all()


        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, stratify=Y, shuffle=True, random_state=42)

        n_splits = 3
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state=0)

        if cv_grid:
            grid_search = GridSearchCV(
                model, cv_grid, "accuracy", cv=cv, refit=False)
            grid_search.fit(X_train, Y_train)
            self.best_params = grid_search.best_params_
            model.set_params(**self.best_params)
        else:
            self.best_params = None


        # Final test with different data
        model.fit(X_train, Y_train)
        Y_true, Y_pred = Y_test, model.predict(X_test)
        self.accuracy = accuracy_score(Y_true, Y_pred, normalize=True)
