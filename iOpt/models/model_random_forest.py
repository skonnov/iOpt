from iOpt.models.model_svm_proba import ModelLinearSVCproba
from sklearn.ensemble import RandomForestClassifier
from iOpt.method.search_data import SearchDataItem
from sklearn.preprocessing import MinMaxScaler

class ModelRandomForestProba(ModelLinearSVCproba):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.init_model()

    def init_model(self):
        self.is_fit = False
        self.svc = RandomForestClassifier(max_depth=2, random_state=0)

    def name(self):
        return "random_forest"
