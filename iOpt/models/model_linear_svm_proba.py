from iOpt.models.model import Model
from sklearn import svm
from iOpt.method.search_data import SearchDataItem

class ModelLinearSVCproba(Model):

    def __init__(self):
        super().__init__()
        self.is_fit = False
        self.svc = svm.SVC(class_weight={1: 98}, probability=True, kernel='linear')  # TODO: use self.parameters.pareto_weight?

    def fit(self, X: list, y: list):
        self.svc.fit(X, y)
        self.is_fit = True

    def calculate_r_ps(self, curr_point: SearchDataItem, left_point: SearchDataItem):
        if not self.is_fit:
            return 0.
        p1 = self.svc.predict_proba([[pt.value for pt in left_point.function_values]])
        p2 = self.svc.predict_proba([[pt.value for pt in curr_point.function_values]])
        r_ps = (p1[0][1] + p2[0][1]) / 2  # [1]?
        return r_ps

    def get_model(self):
        return self.svc