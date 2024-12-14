from iOpt.models.model import Model
import xgboost
from iOpt.method.search_data import SearchDataItem
from sklearn.preprocessing import MinMaxScaler
# import cupy as cp

class ModelXGBoostProba(Model):  # scaled, adjusted weights
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()
        self.init_model()

    def init_model(self):
        self.is_fit = False
        # self.model = xgboost.XGBClassifier(n_estimators=5, max_depth=2, objective='binary:logistic', device="cuda")
        self.model = xgboost.XGBClassifier(n_estimators=5, max_depth=2, objective='binary:logistic')

    def fit(self, X: list, y: list):
        scaled_X = self.scaler.fit_transform(X)
        # scaled_X_cp = cp.array(scaled_X)
        # self.model.fit(scaled_X_cp, y)
        self.model.fit(scaled_X, y)
        self.is_fit = True

    def calculate_dot_characteristic(self, *point):
        p = self.model.predict_proba(self.scaler.transform([point]))
        assert abs(p[0][0] + p[0][1] - 1) < 1e-5
        return p[0][1]

    def calculate_r_ps(self, curr_point: SearchDataItem, left_point: SearchDataItem):
        if not self.is_fit:
            return 0.

        p1 = self.calculate_dot_characteristic(*[pt.value for pt in left_point.function_values])
        p2 = self.calculate_dot_characteristic(*[pt.value for pt in curr_point.function_values])
        # p1 -= 0.5
        # p2 -= 0.5

        r_ps = (p1 + p2) / 2 # [1]?
        return r_ps

    def get_model(self):
        return self.model

    def name(self):
        return "xgboost"
