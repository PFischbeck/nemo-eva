import numpy

from sklearn.model_selection import \
    cross_val_score, GridSearchCV, StratifiedShuffleSplit


class Classification:
    def __init__(self, X, Y, model, cv_grid=None):
        assert numpy.isfinite(X).all().all()

        n_splits = 5
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state=0)

        if cv_grid:
            grid_search = GridSearchCV(
                model, cv_grid, "accuracy", cv=cv, refit=False)
            grid_search.fit(X, Y)
            self.best_params = grid_search.best_params_
            model.set_params(**self.best_params)
        else:
            self.best_params = None


        # Final test with different splits
        cv2 = StratifiedShuffleSplit(n_splits=n_splits, test_size=1/n_splits, random_state=42)
        self.accuracy = cross_val_score(model, X, Y, cv=cv2).mean()
