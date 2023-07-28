"""KMeans分类鸢尾花"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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


def get_dist_euc(point1: np.ndarray, point2: np.ndarray) -> np.float64:
    """求解两点间欧氏距离"""
    return np.sqrt(np.sum(np.square(point1 - point2)))


def rand_init_centroid(dataset: np.ndarray, cluster_num: int) -> np.ndarray:
    """随机初始化质心"""
    n, m = dataset.shape
    centroids = np.zeros((cluster_num, m))
    index_list = random.sample(np.arange(n).tolist(), cluster_num)
    for i in range(cluster_num):
        centroids[i, :] = dataset[index_list[i], :]
    return centroids


def train_KMeans(dataset: np.ndarray, cluster_num: int) -> tuple:
    """
    根据给定数据以及聚类数手动实现KMeans聚类

    Dependency: get_dist_euc, rand_init_centroid

    Return:
        centroids(np.ndarray(cluster_num*feature_num)),
        cluster_belong(list(int)),
        cluster_dist2(list(float))
    """
    # cluster_belong第一列记录归属类别编号，第二列记录该样本距离簇心误差
    record_num = dataset.shape[0]
    cluster_num = min(max(2, cluster_num), record_num)
    cluster_belong = [0 for i in range(record_num)]
    cluster_dist2 = [0 for i in range(record_num)]
    cluster_change = True
    centroids = rand_init_centroid(dataset, cluster_num)
    while cluster_change:
        cluster_change = False
        for i in range(record_num):
            min_cluster_id, min_dist = -1, np.inf
            for j in range(cluster_num):
                dist_tmp = get_dist_euc(dataset[i, :], centroids[j, :])
                if dist_tmp < min_dist:
                    min_cluster_id, min_dist = j, dist_tmp
            if min_cluster_id != cluster_belong[i]:
                cluster_change = True
                cluster_belong[i] = min_cluster_id
            cluster_dist2[i] = min_dist
        for i in range(cluster_num):
            index_list = [idx for idx in range(record_num) if cluster_belong[idx] == i]
            if len(index_list) == 0:
                centroids[i, :] = dataset[int(np.random.uniform(0, record_num)), :]
            else:
                centroids[i, :] = np.mean(dataset[index_list, :], axis=0)
    return centroids, cluster_belong, cluster_dist2


def view_cluster_result(
    data: np.ndarray, cluster_belong: list, centroids: np.ndarray = None
) -> None:
    """聚类结果可视化"""
    iris_feature = ["sepal length", "sepal width", "petal length", "petal width"]
    record_num, cluster_num = data.shape[0], 3
    cluster_contain = [
        [idx for idx in range(record_num) if cluster_belong[idx] == i]
        for i in range(cluster_num)
    ]
    color_list, marker_list = ["r", "g", "b"], ["o", "^", "s"]

    plt.suptitle("Iris Classification via KMeans")
    plt.subplot(1, 2, 1)
    for i in range(cluster_num):
        plt.scatter(
            data[cluster_contain[i], 0],
            data[cluster_contain[i], 1],
            c=[color_list[cluster_belong[idx]] for idx in cluster_contain[i]],
            marker=marker_list[i],
            alpha=0.5,
        )
    plt.legend(["cluster {}".format(i) for i in range(cluster_num)])
    if centroids is not None:
        plt.scatter(
            centroids[:, 0], centroids[:, 1], c=color_list, marker="*", s=60, alpha=0.7
        )
    plt.xlabel(iris_feature[0])
    plt.ylabel(iris_feature[1])
    plt.subplot(1, 2, 2)
    for i in range(cluster_num):
        plt.scatter(
            data[cluster_contain[i], 2],
            data[cluster_contain[i], 3],
            c=[color_list[cluster_belong[idx]] for idx in cluster_contain[i]],
            marker=marker_list[i],
            alpha=0.5,
        )
    plt.legend(["cluster {}".format(i) for i in range(cluster_num)])
    if centroids is not None:
        plt.scatter(
            centroids[:, 2], centroids[:, 3], c=color_list, marker="*", s=60, alpha=0.7
        )
    plt.xlabel(iris_feature[2])
    plt.ylabel(iris_feature[3])
    plt.show()


if __name__ == "__main__":
    # 数据加载
    total_x, total_y = load_iris_data(4)
    cluster_num, record_num = 3, total_x.shape[0]
    randseed = 53
    random.seed(randseed)
    train_x, test_x, train_y, test_y = model_selection.train_test_split(
        total_x, total_y, random_state=randseed, test_size=0.2
    )
    # 自定义KMeans
    # 模型训练
    centroids, cluster_belong, cluster_dist2 = train_KMeans(total_x, 3)
    # 模型评估
    for i in range(3):
        dist2cent = [
            cluster_dist2[idx] for idx in range(record_num) if cluster_belong[idx] == i
        ]
        print("cluster {} mean error: {}".format(i, sum(dist2cent) / len(dist2cent)))
    view_cluster_result(total_x, cluster_belong, centroids)
    # sklearn库KMeans
    model = KMeans(n_clusters=cluster_num)
    model.fit(total_x)
    total_y_pred = model.labels_
    model_centroids = model.cluster_centers_
    view_cluster_result(total_x, total_y_pred, model_centroids)
