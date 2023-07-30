"""FCNN分类宝石图片"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam, SGD
import matplotlib.pyplot as plt

from CustomDataset import CustomDataset, get_origin_data_array

device = "cuda:0" if torch.cuda.is_available() else "cpu"
datatype = torch.float32
pic_resize_shape = (224, 224)  # 图片规格化尺寸


def get_dataloader(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    batch_size: int = 32,
) -> (DataLoader, DataLoader):
    """根据给定数据创建数据加载器"""
    train_loader = DataLoader(
        CustomDataset(X_train, Y_train, datatype=datatype, device=device),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        CustomDataset(X_test, Y_test, datatype=datatype, device=device),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader


class FCNN(nn.Module):
    """自定义全连接网络分类宝石"""

    def __init__(self) -> None:
        """模型搭建"""
        super().__init__()
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(pic_resize_shape[0] * pic_resize_shape[1] * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 25),
        )
        self.to(device=device)

    def forward(self, X):
        """前向传播"""
        X = self.flatten(X)
        return self.seq(X)


def train_single_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn,
    optimizer: Optimizer,
    show_loss: bool = False,
) -> (float, float):
    """模型训练"""
    model.train()
    size = len(data_loader.dataset)
    train_loss, train_accuracy = 0, 0
    for batch, (X, Y_real) in enumerate(data_loader):
        Y_pred = model(X)
        batch_loss = loss_fn(Y_pred, Y_real)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        now_loss = batch_loss.item()
        now_correct = Y_pred.argmax(dim=1).eq(Y_real.argmax(dim=1)).sum().item()
        train_loss += now_loss * len(X)
        train_accuracy += now_correct
        if show_loss and ((batch + 1) % 100 == 0):
            progress = min((batch + 1) * len(X), size)
            print(
                "loss: {:>7f}, accu: {:>7f}, [{:>5d}/{:>5d}]".format(
                    now_loss, now_correct / len(X), progress, size
                )
            )
    else:
        if show_loss and ((batch + 1) % 100 != 0):
            print(
                "loss: {:>7f}, accu: {:>7f}, [{:>5d}/{:>5d}]".format(
                    now_loss, now_correct / len(X), size, size
                )
            )
    train_loss /= size
    train_accuracy /= size
    return train_loss, train_accuracy


def get_test_loss(
    model: nn.Module, data_loader: DataLoader, loss_fn, show_loss: bool = False
) -> (float, float):
    """模型测试"""
    model.eval()
    size = len(data_loader.dataset)
    test_loss, test_accuracy = 0, 0
    with torch.no_grad():
        for X, Y_real in data_loader:
            Y_pred = model(X)
            batch_loss = loss_fn(Y_pred, Y_real)
            test_loss += batch_loss.item() * len(X)
            test_accuracy += Y_pred.argmax(dim=1).eq(Y_real.argmax(dim=1)).sum().item()
        test_loss /= size
        test_accuracy /= size
    if show_loss:
        print(
            "Test Error: loss: {:>7f}, accu: {:>7f}\n".format(test_loss, test_accuracy)
        )
    return test_loss, test_accuracy


def predict(model: nn.Module, X: np.ndarray, norm: bool = True) -> np.ndarray:
    """预测给定输入X的标记"""
    model.eval()
    X = X.astype(np.float32)
    if norm:
        X /= 255.0
    with torch.no_grad():
        Y_pred = model(torch.tensor(X, dtype=datatype, device=device))
    if device != "cpu":
        Y_pred = Y_pred.cpu()
    return Y_pred.numpy()


def view_loss_accu_curve(
    train_loss_list: list,
    valid_loss_list: list,
    train_accu_list: list,
    valid_accu_list: list,
) -> None:
    """绘制训练过程中的loss和accuracy变化曲线"""
    plt.suptitle("gem classification")
    plt.subplot(1, 2, 1)
    plt.xlabel("epoch"), plt.ylabel("loss")
    plt.plot(epoch_list, train_loss_list, "bo-")
    plt.plot(epoch_list, valid_loss_list, "rx-")
    plt.legend(["train", "valid"])
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.xlabel("epoch"), plt.ylabel("accuracy")
    plt.plot(epoch_list, train_accu_list, "bo-")
    plt.plot(epoch_list, valid_accu_list, "rx-")
    plt.legend(["train", "valid"])
    plt.grid()
    plt.show(block=False)
    pass


if __name__ == "__main__":
    print("Using device: {}".format(device))
    dir_path = r".\Chapter4_NeuralNetworkBasic\Lab14_FCNNClassifyingGem\data"
    # 数据加载
    X_total, Y_total, target_names = get_origin_data_array(
        os.path.join(dir_path, "train"), pic_resize_shape
    )
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_total, Y_total, train_size=0.8, random_state=23, stratify=Y_total
    )
    train_loader, valid_loader = get_dataloader(X_train, Y_train, X_valid, Y_valid)
    X_test, Y_test, _ = get_origin_data_array(
        os.path.join(dir_path, "test"), pic_resize_shape, target_names
    )
    # 模型加载/训练
    model_filename = "model_{}x{}.pt".format(pic_resize_shape[0], pic_resize_shape[1])
    model_filepath = os.path.join(dir_path, model_filename)
    model = FCNN()
    train_flag = True
    if os.path.isfile(model_filepath):
        model = torch.load(model_filepath)
        print("finish loading pretrained model")
        keep_on_train = input("keep on training the model or not?(y for yes): ")
        if keep_on_train != "y":
            train_flag = False
    if train_flag:
        epochs = 120
        epoch_list = [i for i in range(1, 1 + epochs)]
        learning_rate = 10**-3
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), learning_rate)
        train_loss_list, valid_loss_list = [], []
        train_accu_list, valid_accu_list = [], []
        for epoch in epoch_list:
            show_loss = False
            if epoch % 10 == 1:
                show_loss = True
                print("--------Epoch {:>3d}--------".format(epoch))
            train_loss, train_accu = train_single_epoch(
                model, train_loader, loss_fn, optimizer, show_loss
            )
            valid_loss, valid_accu = get_test_loss(
                model, valid_loader, loss_fn, show_loss
            )
            train_loss_list.append(train_loss)
            train_accu_list.append(train_accu)
            valid_loss_list.append(valid_loss)
            valid_accu_list.append(valid_accu)
        plt.figure(1)
        view_loss_accu_curve(
            train_loss_list, valid_loss_list, train_accu_list, valid_accu_list
        )
        plt.figure(2)
    # 模型测试
    Y_test_pred = predict(model, X_test)
    np.array([[1]])
    targets_real = [target_names[index] for index in np.argmax(Y_test, axis=1).tolist()]
    targets_pred = [
        target_names[index] for index in np.argmax(Y_test_pred, axis=1).tolist()
    ]
    for i in range(len(targets_pred)):
        plt.subplot(5, 5, i + 1)
        plt.axis("off")
        # cv2读取图像按照bgr,pyplot显示图像按照rgb,则需要调换对应顺序
        plt.imshow(cv2.cvtColor(cv2.resize(X_test[i], (64, 64)), cv2.COLOR_BGR2RGB))
        plt.title(
            "real: {}, pred: {}".format(targets_real[i], targets_pred[i]), fontsize=8
        )
    # 模型保存
    if train_flag:
        plt.show(block=False)
        save_model = input("save the latest trained model or not?(y for yes): ")
        if save_model == "y":
            torch.save(model, model_filepath)
            print("finish saving the model")
    else:
        plt.show()
