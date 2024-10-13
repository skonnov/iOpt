from iOpt.models.model import Model
from sklearn import svm
from iOpt.method.search_data import SearchDataItem
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ModelLinearSVCprobaAdjWeights(Model):
    def __init__(self):
        super().__init__()
        self.is_fit = False
        self.svc = svm.SVC(class_weight={1: 98}, probability=True, kernel='linear', max_iter=10000)  # TODO: use self.parameters.pareto_weight?
        self.scaler = MinMaxScaler()

    def fit(self, X: list, y: list):
        pareto_size = sum(y)
        # print(pareto_size)
        self.svc.set_params(class_weight={0: pareto_size, 1: len(y) - pareto_size})

        scaled_X = self.scaler.fit_transform(X)
        self.svc.fit(scaled_X, y)

        self.is_fit = True

    def calculate_dot_characteristic(self, *point):
        p = self.svc.predict_proba(self.scaler.transform([point]))
        assert abs(p[0][0] + p[0][1] - 1) < 1e-5
        return p[0][1]


    def calculate_r_ps(self, curr_point: SearchDataItem, left_point: SearchDataItem):
        if not self.is_fit:
            return 0.
        p1 = self.calculate_dot_characteristic(*[pt.value for pt in left_point.function_values])
        p2 = self.calculate_dot_characteristic(*[pt.value for pt in curr_point.function_values])

        r_ps = (p1 + p2) / 2# [1]?
        return r_ps

    def get_model(self):
        return self.svc

    def name(self):
        return "linear_svm_proba_adj_weights"

class ModelPolySVCprobaAdjWeights(ModelLinearSVCprobaAdjWeights):
    def __init__(self):
        self.is_fit = False
        self.svc = svm.SVC(class_weight={1: 98}, probability=True, kernel='poly', max_iter=10000)  # TODO: use self.parameters.pareto_weight?
        self.scaler = MinMaxScaler()

    def name(self):
        return "poly_svm_proba_adj_weights"

class ModelRbfSVCprobaAdjWeights(ModelLinearSVCprobaAdjWeights):
    def __init__(self):
        self.is_fit = False
        self.svc = svm.SVC(class_weight={1: 98}, probability=True, kernel='rbf', max_iter=10000)  # TODO: use self.parameters.pareto_weight?
        self.scaler = MinMaxScaler()

    def name(self):
        return "rbf_svm_proba_adj_weights"
