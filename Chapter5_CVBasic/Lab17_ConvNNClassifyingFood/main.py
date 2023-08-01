"""卷积NN分类食物"""

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from DataPreprocess import get_origin_data_array
from PicSet import get_pic_loader
from Model import save_model, load_or_create_model, fit, predict

datatype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
pic_resize_shape = (64, 64)


def draw_curve(train_list: list, test_list: list, ylabel: str) -> None:
    """绘制train/test结果曲线的对比图"""
    plt.xlabel("epoch"), plt.ylabel(ylabel)
    plt.plot([i for i in range(1, len(train_list) + 1)], train_list, "bo-")
    plt.plot([i for i in range(1, len(test_list) + 1)], test_list, "rx-")
    plt.legend(["train", "test"])


if __name__ == "__main__":
    print("Using device: {}".format(device))
    dir_path = r".\Chapter5_CVBasic\Lab17_ConvNNClassifyingFood\data"
    # 数据导入
    X_total, Y_total, target_names = get_origin_data_array(
        dir_path, pic_resize_shape=pic_resize_shape
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_total, Y_total, train_size=0.8, random_state=17, stratify=Y_total
    )
    train_loader, test_loader = get_pic_loader(
        X_train,
        Y_train,
        X_test,
        Y_test,
        batch_size=32,
        datatype=datatype,
        device=device,
    )
    print(X_train[1].shape)
    # 模型加载/训练
    model_filename = "model_{}x{}.pth".format(pic_resize_shape[0], pic_resize_shape[1])
    model_filepath = os.path.join(dir_path, model_filename)
    model = load_or_create_model(model_filepath, pic_resize_shape, device)
    train_flag, keep_train = False, input("keep on training the model?(y for yes): ")
    if keep_train == "y":
        train_flag = True
    if train_flag:
        epochs, learning_rate = 100, 10**-3
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        train_loss_list, train_accu_list, test_loss_list, test_accu_list = fit(
            model,
            epochs,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            verbose_epoch_freq=10,
            verbose_train_batch=False,
            verbose_train_batch_freq=10,
        )
        plt.figure(1)
        plt.suptitle("ConvNN Classify Food Pics")
        plt.subplot(1, 2, 1)
        draw_curve(train_loss_list, test_loss_list, "loss")
        plt.subplot(1, 2, 2)
        draw_curve(train_accu_list, test_accu_list, "accuracy")
        plt.show(block=False)
        plt.figure(2)
    # 模型检验
    sample_list = [31, 37, 107, 211]
    X_sample, Y_sample = X_test[sample_list], Y_test[sample_list]
    Y_sample_pred = predict(
        model, torch.tensor(X_sample, dtype=datatype, device=device)
    )
    for i in range(len(sample_list)):
        plt.subplot(2, 2, i + 1)
        plt.axis("off")
        plt.imshow(X_sample[i].transpose((1, 2, 0)))
        plt.title(
            "real: {}\npred: {}".format(
                target_names[np.argmax(Y_sample[i])],
                target_names[np.argmax(Y_sample_pred[i])],
            ),
            fontsize=9,
        )
    plt.show(block=False)
    # 模型保存
    if input("save the model?(y for yes)") == "y":
        save_model(model, model_filepath)
    pass
