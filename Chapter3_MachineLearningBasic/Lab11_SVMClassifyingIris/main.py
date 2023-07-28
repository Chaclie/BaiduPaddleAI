"""支持向量机分类鸢尾花"""

import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import svm
from sklearn import model_selection
from sklearn.datasets import load_iris


def load_iris_data(feature_num: int = 4) -> tuple:
    """加载鸢尾花数据集,并选择前feature_num个特征"""
    train_set = load_iris()
    if (feature_num <= 0) | (feature_num > 4):
        feature_num = 4
    X = train_set.data[:, :feature_num]
    Y = train_set.target
    return X, Y


def view_hyperplane(model: svm.SVC, X: np.ndarray, Y: np.ndarray) -> None:
    """根据model的预测结果形成可视化决策分区"""
    iris_feature = ["sepal length", "sepal width", "petal length", "petal width"]
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    x1_list, x2_list = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((x1_list.flat, x2_list.flat), axis=1)
    grid_pred = model.predict(grid_test).reshape(x1_list.shape)

    cm_light = mpl.colors.ListedColormap(["#FFA0A0", "#A0FFA0", "#A0A0FF"])
    cm_dark = mpl.colors.ListedColormap(["r", "g", "b"])
    plt.pcolormesh(x1_list, x2_list, grid_pred, cmap=cm_light)
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y), edgecolors="k", s=50, cmap=cm_dark)
    plt.xlabel(iris_feature[0])
    plt.ylabel(iris_feature[1])
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title("Iris Classification via SVM")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # 数据加载
    total_x, total_y = load_iris_data(2)
    train_x, test_x, train_y, test_y = model_selection.train_test_split(
        total_x, total_y, random_state=53, test_size=0.2
    )
    # 模型训练
    model = svm.SVC(C=0.8, kernel="linear", decision_function_shape="ovr")
    model.fit(train_x, train_y.ravel())
    # 模型评估
    print("train pred accuracy: {:.3f}".format(model.score(train_x, train_y)))
    print(" test pred accuracy: {:.3f}".format(model.score(test_x, test_y)))
    # print("decision function:\n", model.decision_function(train_x))  # 表示各样本到各个分割平面的距离
    view_hyperplane(model, total_x, total_y)
    pass
