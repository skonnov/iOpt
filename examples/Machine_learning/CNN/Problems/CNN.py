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
    def __init__(self, channel_count=16, kernel_size=3, num_classes=10, fc_size=128):
        super(ExampleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, channel_count, kernel_size=kernel_size, stride=1, padding=1)  # Input: 3 channels (RGB)
        self.conv2 = nn.Conv2d(channel_count, channel_count * 2, kernel_size=kernel_size, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by half

        # Fully connected layers
        self.fc1 = nn.Linear(channel_count * 2 * 8 * 8, fc_size)  # Adjust input size based on image dimensions
        self.fc2 = nn.Linear(fc_size, num_classes)  # Output layer (num_classes)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for images, labels in data_loader: # проход по всем данным
            # Получение выхода сети на входной пачке изображений
            images = images.to(device)
            labels = labels.to(device)
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
                 fc_size_bound: Dict[str, float]):
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
        self.lower_bound_of_float_variables = np.array([learning_rate_bound['low'], fc_size_bound['low']],
                                                   dtype=np.double)
        self.upper_bound_of_float_variables = np.array([learning_rate_bound['up'],  fc_size_bound['up']],
                                                   dtype=np.double)

        # self.discrete_variable_names.append('number_of_nodes')

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

    def calculateAllFunction(self, point: Point, function_values: np.ndarray(shape=(1), dtype=FunctionValue)):

        learning_rate, fc_size = point.float_variables[0], point.float_variables[1]
        fc_size=int(fc_size)
        batch_size = 4
        num_epochs = 5
        # num_epochs = 1
        train_data_loader, test_data_loader = self._get_data_loaders(batch_size=batch_size)
        print("learning_rate: ", learning_rate)
        print("fc_size: ", fc_size)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", device)
        cnn = ExampleCNN(fc_size=fc_size)
        cnn.to(device)

        self._train(cnn=cnn,
                    train_data_loader=train_data_loader,
                    device=device,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate)

        times = []
        accuracies = []
        infer_repeats = 1
        for _ in range(infer_repeats):
            time1 = time.time()
            test_accuracy = float(get_accuracy(test_data_loader, cnn))
            time2 = time.time()
            times.append(time2 - time1)
            accuracies.append(test_accuracy)
        print(times)
        print(accuracies)
        function_values[0].value = np.mean(times)
        function_values[1].value = -np.mean(test_accuracy)

        print(f"time: {time}, function_values: {function_values}")
        return function_values

