from iOpt.models.model import Model
from iOpt.method.search_data import SearchDataItem
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
import torch.optim as optim
import random as rd
class ModelMLPProba(Model):  # scaled, adjusted weights
    iter = 0
    def __init__(self):
        super().__init__()
        self.num_elems = 200
        self.alpha = 0.1

        self.scaler = MinMaxScaler()
        self.init_model()

    def init_model(self):
        self.is_fit = False
        self.model = MLPClassifier(alpha=self.alpha, hidden_layer_sizes=(self.num_elems, self.num_elems),
                              solver='lbfgs', activation='logistic', max_iter=1000, random_state=42)

    def fit(self, X: list, y: list):
        scaled_X = self.scaler.fit_transform(X)

        # experimental!
        # oversample = RandomOverSampler(sampling_strategy='minority')
        # scaled_X, y = oversample.fit_resample(scaled_X, y)
        # print("Oversampled class distribution:", len(y))
        ModelMLPProba.iter += 1
        print("MLP fit #", ModelMLPProba.iter)

        # scaled_X, y = zip(*rd.shuffle(zip(scaled_X, y)))
        self.model.fit(scaled_X, y)

        self.is_fit = True

    def calculate_dot_characteristic(self, *point) -> float:
        p = self.model.predict_proba([point])
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
        return "MLP"
