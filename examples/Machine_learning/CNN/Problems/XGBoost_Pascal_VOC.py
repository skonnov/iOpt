import numpy as np
import time
import xgboost as xgb
from iOpt import models
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
from torchvision import models
# from thop import profile
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
import os

CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]

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

def build_voc_loaders(
    root: str,
    year: str = "2007",
    batch_size: int = 64,
    num_workers: int = 4,
):
    # Transforms:
    # Even though the model accepts variable HxW, batching requires consistent sizes.
    # Resize to a common size for efficiency; variable size is still OK if batch_size=1.
    img_transform = transforms.Compose([
        transforms.Resize(256),                # keep aspect ratio, shorter side = 256
        transforms.CenterCrop(224),            # or RandomResizedCrop(224) for aug
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225]),
    ])

    train_set = VOCDetection(
        root=root, year=year, image_set="train",
        download=False, transform=img_transform, target_transform=target_to_multihot,
    )
    val_set = VOCDetection(
        root=root, year=year, image_set="val",
        download=False, transform=img_transform, target_transform=target_to_multihot,
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def train_xgb_ovr(X_train: np.ndarray, Y_train: np.ndarray, num_estimators, max_depth, learning_rate=0.05) -> OneVsRestClassifier:
    """
    Trains 20 binary XGBClassifiers (one per class) wrapped in OneVsRestClassifier.
    """
    base = XGBClassifier(
        objective="binary:logistic",
        n_estimators=num_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
        random_state=42,
    )
    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(X_train, Y_train)  # Y_train shape (N, 20), entries {0,1}
    return clf

# -------------------------
# Feature extractor
# -------------------------
class ResNet18Embed(nn.Module):
    """
    Frozen ResNet-18 backbone -> 512-D global embedding per image.
    """
    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = model_dir
        weights_path = os.path.join(self.model_dir, 'resnet18_imagenet1k_v1.pth')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Pretrained weights not found at {weights_path}\n"
                "Please download from: https://download.pytorch.org/models/resnet18-f37072fd.pth\n"
                "And save to the model directory."
            )
        m = models.resnet18(weights=None)
        m.load_state_dict(torch.load(weights_path, map_location='cpu'))
        # m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Drop the final FC, keep avgpool
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # -> (B,512,1,1)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)         # (B,512,1,1)
        return x.flatten(1)          # (B,512)

@torch.no_grad()
def extract_embeddings(loader: DataLoader, device: torch.device, model_dir):
    model = ResNet18Embed(model_dir=model_dir).to(device).eval()
    feats, labels = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        z = model(imgs)                 # (B,512)
        feats.append(z.cpu())
        labels.append(ys.cpu().float())
    X = torch.cat(feats, 0).numpy()     # (N,512)
    Y = torch.cat(labels, 0).numpy()    # (N,20)
    return X, Y


# -------------------------
# Evaluate mAP on val
# -------------------------
def average_precision_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    AP for one class using the 'all-points' interpolation.
    y_true:  (N,) {0,1}
    y_score: (N,) real-valued scores (probabilities or logits)
    Returns AP in [0,1], NaN if no positives.
    """
    y_true = y_true.to(torch.float32)
    order = torch.argsort(y_score, descending=True)
    y_true_sorted = y_true[order]

    tp = torch.cumsum(y_true_sorted, dim=0)
    fp = torch.cumsum(1 - y_true_sorted, dim=0)

    precision = tp / (tp + fp).clamp_min(1e-12)
    num_pos = y_true.sum()
    if num_pos == 0:
        return float('nan')
    recall = tp / num_pos

    # Precision envelope with sentinel endpoints
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = torch.nonzero(mrec[1:] != mrec[:-1]).flatten()
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()
    return ap

def compute_map_from_scores(scores: np.ndarray, targets: np.ndarray):
    """
    scores:  (N, C) real-valued scores (e.g., proba of class=1)
    targets: (N, C) {0,1} multi-hot
    Returns: (mAP, per_class_AP_dict)
    """
    logits_t = torch.from_numpy(scores)
    targets_t = torch.from_numpy(targets).float()
    C = logits_t.shape[1]
    aps = []
    for c in range(C):
        ap = average_precision_torch(targets_t[:, c], logits_t[:, c])
        aps.append(ap)
    aps_t = torch.tensor(aps, dtype=torch.float32)
    if torch.any(~torch.isnan(aps_t)):
        mAP = torch.nanmean(aps_t).item()
    else:
        mAP = float('nan')
    per_class = {cls: float(ap) for cls, ap in zip(CLASS_NAMES, aps)}
    return mAP, per_class

def evaluate_xgb_map(clf: OneVsRestClassifier, X_val: np.ndarray, Y_val: np.ndarray):
    """
    Produces per-class probabilities and computes per-class AP and mAP.
    """
    # OneVsRestClassifier.predict_proba returns list/array of shape (N, C)
    # with positive-class probabilities for each classifier.

    time1 = time.time()
    proba = clf.predict_proba(X_val)          # (N, 20)
    time2 = time.time()
    mAP, per_class = compute_map_from_scores(proba, Y_val)
    return mAP, per_class, time2-time1

class XGBoostPascalVoc(Problem):
    def __init__(self, #x_dataset: np.ndarray, y_dataset: np.ndarray,
                 learning_rate_bound: Dict[str, float],
                 fc_size_bound: Dict[str, float]):
        super(XGBoostPascalVoc, self).__init__()
        self.name = "XGboost"
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

    def calculateAllFunction(self, point: Point, function_values: np.ndarray(shape=(1), dtype=FunctionValue)):
        root = r"D:\\datasets\\VOCtrainval_06-Nov-2007"
        model_dir = r"D:\\Works\\iOpt-scripts"

        num_estimators, max_depth = point.float_variables[0], point.float_variables[1]
        num_estimators = int(num_estimators)
        max_depth = int(max_depth)
        print("num_estimators: ", num_estimators)
        print("max_depth: ", max_depth)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", device)


        # Build loaders
        train_loader, val_loader = build_voc_loaders(root=root, year="2007", batch_size=64, num_workers=4)

        # # Extract embeddings
        X_train, Y_train = extract_embeddings(train_loader, device, model_dir)
        X_val,   Y_val   = extract_embeddings(val_loader, device, model_dir)

        # # Train XGBoost OVR
        clf = train_xgb_ovr(X_train, Y_train, num_estimators=num_estimators, max_depth=max_depth)

        # # Evaluate
        times = []
        accuracies = []
        infer_repeats = 1
        for _ in range(infer_repeats):
            mAP, per_class_ap, inf_time = evaluate_xgb_map(clf, X_val, Y_val)
            times.append(inf_time)
            accuracies.append(mAP)
        print(times)
        print(accuracies)
        function_values[0].value = np.mean(times)
        function_values[1].value = -np.mean(accuracies)

        return function_values

