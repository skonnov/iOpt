import numpy as np
import time
import xgboost as xgb
from iOpt.trial import Point
from iOpt.trial import FunctionValue
from iOpt.problem import Problem
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from thop import profile

class ExampleNet(nn.Module):
    def __init__(self, N_input: int, N_output: int, nodes_count: int):
        super.__init__()
        self.lin1 = nn.Linear(N_input, nodes_count)
        self.lin2 = nn.Linear(nodes_count, N_output)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        # x = torch.relu(self.lin2(x))
        x = self.softmax(x)
        return x

class ExampleCNN(nn.Module):
    def __init__(self, kernel_size=3, num_classes=10, ):
        super(ExampleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1)  # Input: 3 channels (RGB)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by half

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Adjust input size based on image dimensions
        self.fc2 = nn.Linear(128, num_classes)  # Output layer (num_classes)

    def forward(self, x):
        # Conv + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))  # After conv1: (16, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # After conv2: (32, H/4, W/4)

        # Flatten for FC layers
        x = x.view(-1, 32 * 8 * 8)  # Adjust based on final feature map size

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Raw logits (no softmax yet)
        return x

def get_accuracy(data_loader, model):
    tp = 0
    n = 0
    with torch.no_grad():
        for images, labels in data_loader: # проход по всем данным
            # Получение выхода сети на входной пачке изображений

            images = images.to(torch.device("cuda"))
            labels = labels.to(torch.device("cuda"))
            outputs = model(images)
            # Выбор предсказанных меток с максимальной достоверностью.
            # outputs.data - объект типа torch.tensor, двумерный тензор, массив
            # векторов достоверности принадлежности каждому из 10 допустимых классов
            # (размерность 0 - номер изображения в пачке, размерность 1 - номер класса);
            # predicted - объект типа torch.tensor (одномерный тензор меток классов).
            # Выбор максимальных значений выполняется по первой размерности
            _, predicted = torch.max(outputs.data, 1)
            n += labels.size(0) # количество изображений, совпадает с batch_size
            tp += (predicted == labels).sum() # определение количества корректных совпадений
    return tp / n


class CNN(Problem):
    def __init__(self, #x_dataset: np.ndarray, y_dataset: np.ndarray,
                 learning_rate_bound: Dict[str, float],
                 kernel_size: Dict[str, float]):
        super(CNN, self).__init__()
        self.name = "CNN"
        self.dimension = 2 # number of nodes on the first layer, learning rate
        self.number_of_float_variables = 2
        self.number_of_discrete_variables = 0
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        # if x_dataset.shape[0] != y_dataset.shape[0]:
        #     raise ValueError('The input and output sample sizes do not match.')
        # self.x = x_dataset
        self.float_variable_names = np.array(["learning_rate", "nodes_count"], dtype=str)
        self.lower_bound_of_float_variables = np.array([learning_rate_bound['low'], kernel_size['low']],
                                                   dtype=np.double)
        self.upper_bound_of_float_variables = np.array([learning_rate_bound['up'],  kernel_size['up']],
                                                   dtype=np.double)

        # self.discrete_variable_names.append('number_of_nodes')


        # Variables to set objectives. Needed because objectives calculation function is called separately for each
        # objective, but for current problem it is calculated only once after one inference.
        self.test_accuracy = 0.
        self.time = 0.


    def _get_data_loaders(self, batch_size):
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(root = './datacifar', train = True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root = './datacifar', train = False, download=True, transform=transform)

        # Создание объектов для последовательной загрузки пачек
        # из тренировочной и тестовой выборок
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size,
                                                        shuffle = True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size,
                                                    shuffle = False)
        return train_data_loader, test_data_loader

    def _train(self, cnn, train_data_loader, device, num_epochs, learning_rate):
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(cnn.parameters(), lr = learning_rate)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_data_loader):
                images = images.requires_grad_().to(device)
                labels = labels.to(device)
                # Прямой проход
                outputs = cnn(images)
                loss = loss_function(outputs, labels)
                # Обратный проход
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch[{}]: accuracy = {}'.
            format(epoch, get_accuracy(train_data_loader, cnn)))

    # def _get_model_flop(self, model, input_tensor):
    #     macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    #     total_flops = macs * 2  # Convert MACs to FLOPs
    #     return total_flops

    def calculate(self, point: Point, function_value: FunctionValue) -> FunctionValue:
        print("function_value.functionID: ", function_value.functionID)
        if function_value.functionID == 0:
            learning_rate, kernel_size = point.float_variables[0], point.float_variables[1]
            kernel_size = int(kernel_size)

            batch_size = 4
            # num_epochs = 5
            num_epochs = 1
            train_data_loader, test_data_loader = self._get_data_loaders(batch_size=batch_size)
            print("learning_rate: ", learning_rate)
            print("kernel_size: ", kernel_size)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("device: ", device)
            cnn = ExampleCNN()
            cnn.to(device)

            self._train(cnn=cnn,
                        train_data_loader=train_data_loader,
                        device=device,
                        num_epochs=num_epochs,
                        learning_rate=learning_rate)

            time1 = time.time()
            self.test_accuracy = float(get_accuracy(test_data_loader, cnn))
            self.time = time.time() - time1

        if function_value.functionID == 0: # time (TODO: fps)
            function_value.value = self.time
        if function_value.functionID == 1:
            function_value.value = -self.test_accuracy

        print(f"calculate for {function_value.functionID} objective, got {function_value.value} result")
        print(type(function_value.value))
        return function_value

