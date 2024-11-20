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
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        x = nn.functional.relu(self.lin3(x))
        x = self.softmax(x)
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

        # print(f"Using {self.device} device")

        # self.flatten = nn.Flatten()

        self.net = Net(2)
        self.criterion = nn.CrossEntropyLoss(torch.Tensor([0.8, 0.2]).to(torch.float32))  # change weigths?
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        self.scaler = MinMaxScaler()

    def fit(self, X: list, y: list):
        scaled_X = self.scaler.fit_transform(X)

        self.optimizer.zero_grad()

        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i in range(len(scaled_X)):
                outputs = self.net(torch.from_numpy(scaled_X[i]).to(torch.float32))
                loss = self.criterion(outputs, torch.Tensor([y[i], 1-y[i]]).to(torch.float32))
                loss.backward()
                self.optimizer.step()

                # print statistics
                # running_loss += loss.item()
                # if i % 1 == 0:    # print every 2000 mini-batches
                #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                #     running_loss = 0.0

        self.is_fit = True

    def calculate_dot_characteristic(self, *point):
        p = self.net(torch.Tensor(point).to(torch.float32))
        # print("-->", p[0], p[1], "<--")
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
        return "NN"
