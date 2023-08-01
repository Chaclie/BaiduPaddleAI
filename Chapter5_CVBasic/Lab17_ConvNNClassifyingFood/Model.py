"""卷积NN模型搭建"""

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class PicModel(nn.Module):
    """自定义卷积NN"""

    def __init__(self, pic_resize_shape: tuple) -> None:
        super().__init__()
        if pic_resize_shape != (64, 64):
            raise ValueError("pic size and model input size do not match ")
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=6 * 6 * 16, out_features=5),
        )

    def forward(self, X):
        return self.seq(X)


def train_for_single_epoch(
    model: nn.Module,
    pic_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    verbose_epoch: bool = True,
    verbose_batch: bool = False,
    verbose_batch_freq: int = 10,
) -> (float, float):
    if verbose_batch:
        verbose_epoch = True
    if verbose_batch_freq <= 0:
        verbose_batch_freq = 10
    model.train()
    size = len(pic_loader.dataset)
    train_loss, train_crct = 0, 0
    for batch, (X, Y_real) in enumerate(pic_loader):
        Y_pred = model(X)
        batch_loss = loss_fn(Y_pred, Y_real)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss = batch_loss.item()
        tot_correct = Y_pred.argmax(dim=1).eq(Y_real.argmax(dim=1)).sum().item()
        train_loss += avg_loss * len(X)
        train_crct += tot_correct
        if verbose_batch and ((batch + 1) % verbose_batch_freq == 0):
            progress = max((batch + 1) * len(X), size)
            print(
                "Batch {}: loss={:5f}, accu={:5f}, progress=[{:5d}/{:5d}]".format(
                    batch, avg_loss, tot_correct / len(X), progress, size
                )
            )
    else:
        if verbose_epoch:
            if not verbose_batch or ((batch + 1) % verbose_batch_freq != 0):
                print(
                    "Batch {}: loss={:5f}, accu={:5f}, progress=[{:5d}/{:5d}]".format(
                        batch, avg_loss, tot_correct / len(X), size, size
                    )
                )
    return train_loss / size, train_crct / size


def test(
    model: nn.Module, pic_loader: DataLoader, loss_fn, verbose: bool = True
) -> (float, float):
    model.eval()
    size = len(pic_loader.dataset)
    test_loss, test_crct = 0, 0
    with torch.no_grad():
        for X, Y_real in pic_loader:
            Y_pred = model(X)
            batch_loss = loss_fn(Y_pred, Y_real)
            avg_loss = batch_loss.item()
            tot_correct = Y_pred.argmax(dim=1).eq(Y_real.argmax(dim=1)).sum().item()
            test_loss += avg_loss * len(X)
            test_crct += tot_correct
    if verbose:
        print(
            "Test: loss={:5f}, accu={:5f}\n".format(test_loss / size, test_crct / size)
        )
    return test_loss / size, test_crct / size


def fit(
    model: nn.Module,
    epochs: int,
    train_pic_loader: DataLoader,
    test_pic_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    verbose_epoch_freq: int = 10,
    verbose_train_batch: bool = False,
    verbose_train_batch_freq: int = 10,
) -> (list, list, list, list):
    """
    Return: train_loss_list, train_accu_list, test_loss_list, test_accu_list
    """
    train_loss_list, train_accu_list = [], []
    test_loss_list, test_accu_list = [], []
    for epoch in range(epochs):
        view_verbose = False
        if epoch % verbose_epoch_freq == 0:
            print("------Epoch {:3d}------".format(epoch))
            view_verbose = True
        train_loss, train_accu = train_for_single_epoch(
            model,
            train_pic_loader,
            loss_fn,
            optimizer,
            view_verbose,
            view_verbose and verbose_train_batch,
            verbose_train_batch_freq,
        )
        test_loss, test_accu = test(model, test_pic_loader, loss_fn, view_verbose)
        train_loss_list.append(train_loss), train_accu_list.append(train_accu)
        test_loss_list.append(test_loss), test_accu_list.append(test_accu)
    return train_loss_list, train_accu_list, test_loss_list, test_accu_list


def predict(model: nn.Module, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Y_pred = model(X)
        Y_pred = Y_pred.cpu().numpy()
    return Y_pred


def save_model(model: nn.Module, file_path: str) -> None:
    torch.save(model, file_path)
    print("finish saving model")


def load_or_create_model(
    file_path: str, pic_resize_shape: tuple, device: str = "cpu"
) -> PicModel:
    model = PicModel(pic_resize_shape)
    if os.path.isfile(file_path):
        model = torch.load(file_path)
        print("finish loading model")
    model.to(device=device)
    return model


if __name__ == "__main__":
    pass
