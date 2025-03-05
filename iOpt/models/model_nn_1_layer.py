from iOpt.models.model import Model
from iOpt.method.search_data import SearchDataItem
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Net1Layer(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        self.lin1 = nn.Linear(N, 100)
        self.lin3 = nn.Linear(100, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = self.lin3(x)
        x = self.softmax(x)
        return x


class ModelNNProba1Layer(Model):  # scaled, adjusted weights
    iter = 0
    cnt0 = 0
    cnt1 = 0
    def __init__(self, weights, input_cnt=2):
        super().__init__()
        self.is_fit = False
        # self.model = xgboost.XGBClassifier(n_estimators=5, max_depth=2, objective='binary:logistic', device="cuda")
        self.device = "cpu"
        # self.device = (
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else "mps"
        #     if torch.backends.mps.is_available()
        #     else "cpu")

        # print(f"Using {self.device} device")

        self.weights = weights
        self.input_cnt = input_cnt

        self.scaler = MinMaxScaler()
        self.init_model()

    def init_model(self):
        self.is_fit = False
        self.net = Net1Layer(self.input_cnt)
        self.criterion = nn.CrossEntropyLoss(torch.FloatTensor(self.weights))
        # self.criterion = nn.CrossEntropyLoss(torch.Tensor([0.2, 0.8]).to(torch.float32))  # change weigths?
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    def fit(self, X: list, y: list):
        # print("0s:", ModelNNProba1Layer.cnt0, ", 1s: ", ModelNNProba1Layer.cnt1)
        scaled_X = self.scaler.fit_transform(X)

        scaled_X = torch.FloatTensor(scaled_X)
        y = torch.LongTensor(y)
        dataset = TensorDataset(scaled_X, y)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()  # reset gradients
                outputs = self.net(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                # if i % 1 == 0:    # print every 2000 mini-batches
                #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                #     running_loss = 0.0
            # print(f"Epoch {epoch + 1}/{20}, Loss: {running_loss / len(data_loader):.4f}")
        ModelNNProba1Layer.iter += 1
        # print("NN fit #", ModelNNProba1Layer.iter)
        self.is_fit = True
        ModelNNProba1Layer.cnt0 = 0
        ModelNNProba1Layer.cnt1 = 0

    def calculate_dot_characteristic(self, *point) -> float:
        self.net.eval()
        p = self.net(torch.FloatTensor(self.scaler.transform([point])))
        # print(torch.FloatTensor(self.scaler.transform([point])), "<_-_-_-_-_-", p, point, self.scaler.transform([point]))
        if float(p[0, 1].item()) >= 0.5:
            ModelNNProba1Layer.cnt1 += 1
        else:
            ModelNNProba1Layer.cnt0 += 1
        # print("-->", p[0, 1].item(), "<--")
        return float(p[0, 1].item())

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
        return "NN_1_layer"
