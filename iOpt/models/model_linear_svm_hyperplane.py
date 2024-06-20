from iOpt.models.model import Model
from sklearn import svm
from iOpt.method.search_data import SearchDataItem
from sklearn.preprocessing import MinMaxScaler

class ModelLinearSVChyperplane(Model):

    def __init__(self):
        super().__init__()
        self.is_fit = False
        # self.svc = svm.LinearSVC(class_weight={1: 98}, dual="auto")
        self.svc = svm.SVC(class_weight={1: 98}, probability=True, kernel='linear', max_iter=10000)  # TODO: use self.parameters.pareto_weight?
        self.d_min = 0
        self.d_max = 0
        self.scaler = MinMaxScaler()

    def fit(self, X: list, y: list):
        scaled_X = self.scaler.fit_transform(X)
        self.svc.fit(scaled_X, y)
        
        d = self.svc.decision_function(scaled_X)  # need to divide the function values
                                                       # by the norm of the weight vector (coef_) (in case of decision_function_shape=’ovo’)?
        self.d_min = min(d)
        self.d_max = max(d)
        self.is_fit = True

    def calculate_dot_characteristic(self, *point):
        d = self.svc.decision_function(self.scaler.transform([point]))
        return d

    def calculate_r_ps(self, curr_point: SearchDataItem, left_point: SearchDataItem):
        if not self.is_fit:
            return 0.
        d1 = self.calculate_dot_characteristic(*[pt.value for pt in left_point.function_values])
        if d1 < 0:
            d1 = -d1 / self.d_min
        else:
            d1 = d1 / self.d_max
        d2 = self.calculate_dot_characteristic(*[pt.value for pt in curr_point.function_values])
        if d2 < 0:
            d2 = -d2 / self.d_min
        else:
            d2 = d2 / self.d_max

        r_ps = d1 + d2
        return r_ps

    def get_model(self):
        return self.svc