from iOpt.models.model_linear_svm_proba import ModelLinearSVCproba
from sklearn.ensemble import RandomForestClassifier
from iOpt.method.search_data import SearchDataItem

class ModelRandomForestProba(ModelLinearSVCproba):
    def __init__(self):
        self.is_fit = False
        self.svc = RandomForestClassifier(max_depth=2, random_state=0)
