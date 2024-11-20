from iOpt.models.model import Model
from iOpt.method.search_data import SearchDataItem
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        self.lin1 = nn.Linear(N, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 2)

    def forward(self, x):
        x = nn.ReLU(self.lin1(x))
        x = nn.ReLU(self.lin2(x))
        x = nn.ReLU(self.lin3(x))
        return x


class ModelNNProba(Model):  # scaled, adjusted weights
    def __init__(self):
        super().__init__()
        self.is_fit = False
        # self.model = xgboost.XGBClassifier(n_estimators=5, max_depth=2, objective='binary:logistic', device="cuda")
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu")

        print(f"Using {self.device} device")

        # self.flatten = nn.Flatten()

        self.net = Net(2)
        self.criterion = nn.CrossEntropyLoss(Tensor(0.8, 0.))
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def fit(self, X: list, y: list):
        scaled_X = self.scaler.fit_transform(X)

        self.optimizer.zero_grad()

        for i in range(len(scaled_X)):
            outputs = self.net(scaled_X[i])
            loss = self.criterion(outputs, y[i])
            loss.backward()
            self.optimizer.step()

        self.is_fit = True

    def calculate_dot_characteristic(self, *point):
        data = self.flatten([point])
        data = self.linear_relu_stack(data)

        p = self.net(point)
        print(p[0])
        return p[0]

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
