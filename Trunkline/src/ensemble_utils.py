class EnsembleWrapper:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def predict(self, X, return_std=False):
        base_preds = [model.predict(X).reshape(-1, 1) for model in self.base_models]
        Z = np.hstack(base_preds)
        y_pred = self.meta_model.predict(Z)

        if return_std:
            if hasattr(self.meta_model, "coef_"):
                weights = self.meta_model.coef_.reshape(-1, 1)
                base_stds = []

                for model in self.base_models:
                    if hasattr(model, "estimators_"):
                        all_tree_preds = np.array([t.predict(X) for t in model.estimators_])
                        std = np.std(all_tree_preds, axis=0)
                    elif hasattr(model, "predict") and "return_std" in model.predict.__code__.co_varnames:
                        _, std = model.predict(X, return_std=True)
                    else:
                        std = np.zeros(X.shape[0])
                    base_stds.append(std.reshape(-1, 1))

                base_stds = np.hstack(base_stds)
                ensemble_var = np.sum((weights.T ** 2) * (base_stds ** 2), axis=1)
                ensemble_std = np.sqrt(ensemble_var)
                return y_pred, ensemble_std

            else:
                return y_pred, np.zeros_like(y_pred)
        else:
            return y_pred