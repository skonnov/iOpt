from iOpt.models.model import Model
import xgboost
from iOpt.method.search_data import SearchDataItem

class ModelXGBoostProba(Model):
    def __init__(self):
        super().__init__()
        self.is_fit = False
        self.svc = xgboost.XGBClassifier(n_estimators=5, max_depth=2, learning_rate=1, objective='binary:logistic')

    def fit(self, X: list, y: list):
        self.svc.fit(X, y)
        self.is_fit = True

    def calculate_r_ps(self, curr_point: SearchDataItem, left_point: SearchDataItem):
        if not self.is_fit:
            return 0.
        p1 = self.svc.predict_proba([[pt.value for pt in left_point.function_values]])
        p2 = self.svc.predict_proba([[pt.value for pt in curr_point.function_values]])
        assert abs(p1[0][0] + p1[0][1] - 1) < 1e-5
        assert abs(p2[0][0] + p2[0][1] - 1) < 1e-5
        p1 -= 0.5
        p2 -= 0.5

        r_ps = (p1[0][1] + p2[0][1]) # [1]?
        return r_ps

    def get_model(self):
        return self.svc