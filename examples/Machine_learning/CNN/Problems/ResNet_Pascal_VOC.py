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
from torchvision import transforms as T
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision import models
# from thop import profile


CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]
NAME_TO_IDX = {n:i for i,n in enumerate(CLASS_NAMES)}

# --- Helper: AP for a single class (all-points / precision-envelope integration) ---
@torch.no_grad()
def average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    AP for one class using 'all-points' interpolation.
    y_true:  (N,) {0,1} float/bool
    y_score: (N,) real-valued scores (logits or probabilities)
    Returns AP in [0,1]; NaN if there are no positives.
    """
    y_true = y_true.to(torch.float32)
    order = torch.argsort(y_score, descending=True)
    y_sorted = y_true[order]

    tp = torch.cumsum(y_sorted, dim=0)
    fp = torch.cumsum(1 - y_sorted, dim=0)

    precision = tp / (tp + fp).clamp_min(1e-12)
    num_pos = y_true.sum()
    if num_pos == 0:
        return float('nan')
    recall = tp / num_pos

    # Precision envelope with sentinel endpoints
    device = recall.device
    mrec = torch.cat([torch.tensor([0.0], device=device), recall, torch.tensor([1.0], device=device)])
    mpre = torch.cat([torch.tensor([0.0], device=device), precision, torch.tensor([0.0], device=device)])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = torch.nonzero(mrec[1:] != mrec[:-1]).flatten()
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
    return ap

@torch.no_grad()
def evaluate_map_multilabel(
    model: nn.Module,
    val_loader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Runs inference on val set and computes per-class AP and mAP.
    Uses raw logits (ranking is identical to probabilities).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_logits, all_targets = [], []
    time1 = time.time()
    for imgs, ys in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)                    # (B, C)
        all_logits.append(logits.detach().cpu())
        all_targets.append(ys.detach().cpu().float())
    time2 = time.time()

    logits = torch.cat(all_logits, dim=0)      # (N, C)
    targets = torch.cat(all_targets, dim=0)    # (N, C)
    C = logits.shape[1]

    aps = []
    for c in range(C):
        ap_c = average_precision(targets[:, c], logits[:, c])
        aps.append(ap_c)

    aps_t = torch.tensor(aps, dtype=torch.float32)
    mAP = torch.nanmean(aps_t).item() if torch.any(~torch.isnan(aps_t)) else float('nan')

    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]
    per_class = {name: float(ap) for name, ap in zip(class_names, aps)}
    return mAP, per_class, time2-time1

def target_to_multihot(tgt: dict) -> torch.Tensor:
    ann = tgt["annotation"]
    objs = ann.get("object", None)
    y = torch.zeros(len(CLASS_NAMES), dtype=torch.float32)
    if objs is None:
        return y
    if isinstance(objs, dict):
        objs = [objs]
    for obj in objs:
        cls = obj.get("name", None)
        if cls in NAME_TO_IDX:
            y[NAME_TO_IDX[cls]] = 1.0
    return y

class OfflineModelLoader:
    def __init__(self, model_dir: str = './pretrained_models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def ensure_resnet18_weights(self):
        """Check if weights exist, provide instructions if not"""
        weights_path = os.path.join(self.model_dir, 'resnet18_imagenet1k_v1.pth')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Pretrained weights not found at {weights_path}\n"
                "Please download from: https://download.pytorch.org/models/resnet18-f37072fd.pth\n"
                "And save to the model directory."
            )
        return weights_path

    def build_resnet18_multilabel(self, num_classes: int = 20) -> nn.Module:
        weights_path = self.ensure_resnet18_weights()

        # Build model
        m = models.resnet18(weights=None)
        m.load_state_dict(torch.load(weights_path, map_location='cpu'))

        # Modify for multilabel classification
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)

        return m

@torch.no_grad()
def compute_pos_weight(train_loader: DataLoader, num_classes: int = 20) -> torch.Tensor:
    """
    pos_weight[c] = (#negatives for class c) / (#positives for class c).
    """
    pos = torch.zeros(num_classes, dtype=torch.float64)
    total = 0
    for _, y in train_loader:
        y = y.to(torch.float32)
        pos += y.sum(dim=0).to(torch.float64)
        total += y.shape[0]
    neg = total - pos
    pos = pos.clamp_min(1.0)  # avoid div by zero if a class has 0 positives
    return (neg / pos).to(torch.float32)

class ResNetPascalVoc(Problem):
    def __init__(self, #x_dataset: np.ndarray, y_dataset: np.ndarray,
                 learning_rate_bound: Dict[str, float],
                 fc_size_bound: Dict[str, float]):
        super(ResNetPascalVoc, self).__init__()
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
        train_tf = T.Compose([
            T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(3/4, 4/3)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        val_tf = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        root="D:\\datasets\\VOCtrainval_06-Nov-2007"
        year="2007"
        num_workers=4
        train_set = VOCDetection(root=root, year=year, image_set="train",
                                download=False, transform=train_tf,
                                target_transform=target_to_multihot)   # <-- no lambda
        val_set   = VOCDetection(root=root, year=year, image_set="val",
                                download=False, transform=val_tf,
                                target_transform=target_to_multihot)   # <-- no lambda

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader


    def _train(self, model, train_data_loader, val_data_loader, device, num_epochs, learning_rate):
        # BCEWithLogitsLoss is appropriate for multi-label; you can tune pos_weight for class imbalance.
        pos_weight = compute_pos_weight(train_data_loader, num_classes=20).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_data_loader:
                images, labels = images.to(device), labels.to(device)

                logits = model(images)               # (B, 20)
                loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            mAP, per_cls, inf_time = evaluate_map_multilabel(model, val_data_loader, device, class_names=CLASS_NAMES)
            print("epoch: ", epoch, ", mAP: ", mAP, ", time: ", inf_time)

    def calculateAllFunction(self, point: Point, function_values: np.ndarray(shape=(1), dtype=FunctionValue)):
        learning_rate, batch_size = point.float_variables[0], point.float_variables[1]
        batch_size = 2**(int(batch_size))
        num_epochs = 5
        # num_epochs = 1
        train_data_loader, val_data_loader = self._get_data_loaders(batch_size=batch_size)
        print("learning_rate: ", learning_rate)
        print("batch_size: ", batch_size)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", device)

        # Usage
        loader = OfflineModelLoader("D:\\Works\\iOpt-scripts")
        model = loader.build_resnet18_multilabel(num_classes=20)
        model.to(device)

        self._train(model=model,
                    train_data_loader=train_data_loader,
                    val_data_loader=val_data_loader,
                    device=device,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate)

        times = []
        accuracies = []
        infer_repeats = 1
        for _ in range(infer_repeats):
            mAP, per_cls, inf_time = evaluate_map_multilabel(model, val_data_loader, device, class_names=CLASS_NAMES)
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

