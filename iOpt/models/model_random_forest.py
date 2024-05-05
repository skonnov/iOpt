from iOpt.models.model_linear_svm_proba import ModelLinearSVCproba
from sklearn.ensemble import RandomForestClassifier
from iOpt.method.search_data import SearchDataItem
from sklearn.preprocessing import MinMaxScaler

class ModelRandomForestProba(ModelLinearSVCproba):
    def __init__(self):
        self.is_fit = False
        self.svc = RandomForestClassifier(max_depth=2, random_state=0)
        self.scaler = MinMaxScaler()
