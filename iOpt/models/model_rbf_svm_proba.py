from iOpt.models.model_linear_svm_proba import ModelLinearSVCproba
from iOpt.method.search_data import SearchDataItem
from sklearn import svm

class ModelRbfSVCproba(ModelLinearSVCproba):
    def __init__(self):
        self.is_fit = False
        self.svc = svm.SVC(class_weight={1: 98}, probability=True, kernel='rbf')  # TODO: use self.parameters.pareto_weight?
