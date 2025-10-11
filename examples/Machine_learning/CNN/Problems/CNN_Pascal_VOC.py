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
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
# from thop import profile


CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]

class ExampleCNN(nn.Module):
    def __init__(self, channel_count=16, kernel_size=3, num_classes=20, fc_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channel_count, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel_count, channel_count * 2, kernel_size=kernel_size, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap  = nn.AdaptiveAvgPool2d(1)     # size-agnostic
        self.fc1  = nn.Linear(channel_count * 2, fc_size)
        self.fc2  = nn.Linear(fc_size, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        # return self.softmax(x)  # logits

# --- Helper: AP for a single class (all-points / precision-envelope integration) ---
def average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    y_true:  (N,) {0,1}
    y_score: (N,) logits or scores
    returns AP in [0,1]; NaN if there are no positives.
    """

    print("average_precision: ", y_score, y_true)
    y_true = y_true.to(torch.float32)
    # Sort by score descending
    order = torch.argsort(y_score, descending=True)
    y_true_sorted = y_true[order]

    tp = torch.cumsum(y_true_sorted, dim=0)
    fp = torch.cumsum(1 - y_true_sorted, dim=0)
    precision = tp / (tp + fp).clamp_min(1e-12)
    num_pos = y_true.sum()
    if num_pos == 0:
        return float('nan')
    recall = tp / num_pos

    # Precision envelope
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = torch.nonzero(mrec[1:] != mrec[:-1]).flatten()
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
    return ap

@torch.no_grad()
def evaluate_map_multilabel(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Runs inference on the validation set and computes mAP for multi-label classification.

    Returns:
        mAP: float in [0,1], mean of per-class APs (ignores classes with no positives).

        per_class_ap: dict {class_name: AP in [0,1]} (NaN if no positives in val).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_logits, all_targets = [], []

    time1 = time.time()
    for images, targets in val_loader:
        images = images.to(device)
        logits = model(images)                    # (B, C), raw scores (logits)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu().float())
    time2 = time.time()

    logits = torch.cat(all_logits, dim=0)        # (N, C)
    targets = torch.cat(all_targets, dim=0)      # (N, C)
    N, C = logits.shape

    # --- Compute AP per class and mAP ---
    aps: List[float] = []
    for c in range(C):
        ap_c = average_precision(targets[:, c], logits[:, c])
        aps.append(ap_c)

    aps_t = torch.tensor(aps, dtype=torch.float32)
    if torch.any(~torch.isnan(aps_t)):
        mAP = torch.nanmean(aps_t).item()
    else:
        mAP = float('nan')

    # Per-class dict
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]
    per_class_ap: Dict[str, float] = {name: float(ap) for name, ap in zip(class_names, aps)}

    return mAP, per_class_ap, time2-time1

def target_to_multihot(target):
    """
    VOCDetection returns an annotation dict. We convert it to a 20-dim multi-hot.
    """
    # ----- PASCAL VOC as multi-label classification -----

    name_to_idx = {n:i for i,n in enumerate(CLASS_NAMES)}

    objs = target["annotation"]["object"]
    if isinstance(objs, dict):  # single object edge case
        objs = [objs]
    y = torch.zeros(len(CLASS_NAMES), dtype=torch.float32)
    for obj in objs:
        cls = obj["name"]
        if cls in name_to_idx:
            y[name_to_idx[cls]] = 1.0
    return y

class CNNPascalVoc(Problem):
    def __init__(self, #x_dataset: np.ndarray, y_dataset: np.ndarray,
                 learning_rate_bound: Dict[str, float],
                 fc_size_bound: Dict[str, float]):
        super(CNNPascalVoc, self).__init__()
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
        img_transform = transforms.Compose([
            transforms.Resize(256),                # keep aspect ratio, shorter side = 256
            transforms.CenterCrop(224),            # or RandomResizedCrop(224) for aug
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                std=[0.229,0.224,0.225]),
        ])
        train_set = VOCDetection(
            root="D:\\datasets\\VOCtrainval_06-Nov-2007", year="2007", image_set="train",
            download=False, transform=img_transform, target_transform=target_to_multihot
        )
        val_set = VOCDetection(
            root="D:\\datasets\\VOCtrainval_06-Nov-2007", year="2007", image_set="val",
            download=False, transform=img_transform, target_transform=target_to_multihot
        )

        train_data_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_data_loader   = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        return train_data_loader, val_data_loader

    def _train(self, cnn, train_data_loader, device, num_epochs, learning_rate):
        # BCEWithLogitsLoss is appropriate for multi-label; you can tune pos_weight for class imbalance.
        cnn.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            for images, labels in train_data_loader:
                images, labels = images.to(device), labels.to(device)

                logits = cnn(images)               # (B, 20)
                loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    def calculateAllFunction(self, point: Point, function_values: np.ndarray(shape=(1), dtype=FunctionValue)):
        learning_rate, fc_size = point.float_variables[0], point.float_variables[1]
        fc_size=int(fc_size)
        batch_size = 4
        num_epochs = 5
        # num_epochs = 1
        train_data_loader, val_data_loader = self._get_data_loaders(batch_size=batch_size)
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
            mAP, per_cls, inf_time = evaluate_map_multilabel(cnn, val_data_loader, device, class_names=CLASS_NAMES)
            print(f"mAP: {mAP:.4f}")
            for k, v in per_cls.items():
                print(f"{k:>12s}: {v:.4f}")
            times.append(inf_time)
            accuracies.append(mAP)
        print(times)
        print(accuracies)
        function_values[0].value = np.mean(times)
        function_values[1].value = -np.mean(accuracies)

        return function_values

