from iOpt.models.model import Model
import xgboost
from iOpt.method.search_data import SearchDataItem
from sklearn.preprocessing import MinMaxScaler

class ModelXGBoostProba(Model):  # scaled, adjusted weights
    def __init__(self):
        super().__init__()
        self.is_fit = False
        self.model = xgboost.XGBClassifier(n_estimators=5, max_depth=2, learning_rate=1, objective='binary:logistic')
        self.scaler = MinMaxScaler()

    def fit(self, X: list, y: list):
        self.model.fit(X, y)

        scaled_X = self.scaler.fit_transform(X)
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
