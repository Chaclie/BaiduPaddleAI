"""全连接网络预测房价"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from CustomDataset import CustomDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
datatype = torch.float32


def get_dataloader(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    batch_size: int = 32,
    feature_transform=None,
    target_transform=None,
) -> (DataLoader, DataLoader):
    """根据训练集和测试集创建对应的loader"""
    train_dataset = CustomDataset(
        feature_matrix=X_train,
        target_matrix=Y_train,
        feature_transform=feature_transform,
        target_transform=target_transform,
        datatype=datatype,
        device=device,
    )
    test_dataset = CustomDataset(
        feature_matrix=X_test,
        target_matrix=Y_test,
        feature_transform=feature_transform,
        target_transform=target_transform,
        datatype=datatype,
        device=device,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


class FullConnectNN(nn.Module):
    """自定义全连接网络"""

    def __init__(self) -> None:
        """构建模型网络"""
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        """前向传播"""
        x = self.linear_relu_stack(x)
        return x


def train_loop(
    dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer
) -> float:
    """从dataloader加载数据,以loss_fn为损失函数,利用optimizer优化器训练model"""
    model.train()
    size = len(dataloader.dataset)
    train_loss = torch.tensor(0, dtype=datatype, device=device)
    for batch, (X, Y_real) in enumerate(dataloader):
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y_real)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item() * len(X)
        if (batch + 1) % 100 == 0:
            loss, progress = loss.item(), (batch + 1) * len(X)
            print("loss: {:>7f} [{:>5d}/{:>5d}]".format(loss, progress, size))
    else:
        if (batch + 1) % 100 != 0:
            loss, progress = loss.item(), size
            print("loss: {:>7f} [{:>5d}/{:>5d}]".format(loss, progress, size))
    train_loss /= size
    return train_loss.cpu().item()


def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn) -> float:
    """从dataloader加载数据,以loss_fn为损失函数,计算model的平均误差"""
    model.eval()
    size = len(dataloader.dataset)
    test_loss = torch.tensor(0, dtype=datatype, device=device)
    with torch.no_grad():
        for X, Y_real in dataloader:
            Y_pred = model(X)
            test_loss += loss_fn(Y_pred, Y_real) * len(X)
    test_loss /= size
    print("Test Error: avg loss: {:>8f}\n".format(test_loss))
    return test_loss.cpu().item()


def predict(X: np.ndarray, model: nn.Module) -> np.ndarray:
    """预测给定输入data的标记"""
    model.eval()
    with torch.no_grad():
        Y_pred = model(torch.tensor(X.astype(np.float32)).to(device))
    if device != "cpu":
        Y_pred = Y_pred.cpu()
    return Y_pred.numpy()


def view_infer_result(
    ground_truths: list | np.ndarray, infer_results: list | np.ndarray, blocked=True
) -> None:
    """根据真实值和预测值绘制评估图"""
    min_val = min(min(ground_truths), min(infer_results))
    max_val = max(max(ground_truths), max(infer_results))
    dist = max_val - min_val
    if not dist:
        dist = 1.0
    extend_ratio = 0.05
    left_end, right_end = min_val - dist * extend_ratio, max_val + dist * extend_ratio
    plt.rcParams["font.sans-serif"] = ["SimSun"]  # 选择宋体或其他中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 用于解决负号显示为方块的问题
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title("Boston房价预测 via FCNN")
    plt.plot([left_end, right_end], [left_end, right_end], "g-")
    plt.scatter(ground_truths, infer_results, marker="o", c="r", alpha=0.5)
    plt.show(block=blocked)


if __name__ == "__main__":
    print("Using device: {}".format(device))
    dir_path = (
        "./Chapter4_NeuralNetworkBasic/Lab13_FullConnectNNPredictingHousePrice/data"
    )
    # 数据加载
    housing_df = pd.read_csv(os.path.join(dir_path, "housing.csv"))
    X_total = housing_df.iloc[:, :-1].values
    Y_total = housing_df["MEDV"].values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_total, Y_total, train_size=0.8, random_state=17
    )
    train_loader, test_loader = get_dataloader(X_train, Y_train, X_test, Y_test)
    # 模型加载/训练
    model_filepath = os.path.join(dir_path, "model.pth")
    model = FullConnectNN().to(device)
    train_flag = True
    if os.path.isfile(model_filepath):
        model = torch.load(model_filepath)
        model = model.to(device)
        keep_on_train = input("keep on training the model or not?(y for yes):")
        if keep_on_train != "y":
            train_flag = False
    if train_flag:
        epochs = 15
        learning_rate = 1e-3
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loss_list, test_loss_list = [], []
        epoch_list = [i for i in range(1, epochs + 1)]
        for epoch in epoch_list:
            print("Epoch {}\n-----------------------------".format(epoch))
            train_loss = train_loop(train_loader, model, loss_fn, optimizer)
            test_loss = test_loop(test_loader, model, loss_fn)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
    # 模型评估/保存
    if train_flag:
        # 由于此数据集数据量较少,可能导致数据划分不完全分布相似,进而导致test_loss小于train_loss的情况
        plt.figure(1)
        plt.plot(epoch_list, train_loss_list, "bo-")
        plt.plot(epoch_list, test_loss_list, "rx-")
        plt.legend(["train", "test"])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Boston Housing Price Predicting via FCNN")
        plt.grid()
        plt.show(block=False)
        plt.figure(2)
        view_infer_result(Y_total, predict(X_total, model), False)
        model_save_flag = input("save the model or not?(y for yes):")
        if model_save_flag == "y":
            torch.save(model, model_filepath)
            print("finish saving the model")
    else:
        view_infer_result(Y_total, predict(X_total, model))
    pass
