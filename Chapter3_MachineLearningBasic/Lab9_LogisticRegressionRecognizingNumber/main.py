"""逻辑回归识别数字"""

import os, time, random, struct
from array import array as pyarray
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


def load_mnist(image_file_name: str, label_file_name: str, dir_path: str) -> tuple:
    """
    从dir_path目录加载指定的mnist图像文件和标签文件;
    具体mnist文件的压缩包下载和格式解读见官网"http://yann.lecun.com/exdb/mnist/"

    Return:
    images(np.ndarray:legal_record_num*fig_size uint8),
    labels(np.ndarray:legal_record_num*1 int8)
    """
    with open(os.path.join(dir_path, image_file_name), "rb") as fp:
        # magic_number用于标识文件格式,2051标识图像文件,2049标识标签文件
        # unpack的format参数中">"指明大端字节序,I表示4字节无符号整数,i表示有符号,
        # b-byte,h-half,l-long,大写与I和i的关系类似;
        # f-float,d-double,s-string(需指明长度)
        magic_number, record_num = struct.unpack(">ii", fp.read(8))
        if magic_number != 2051:
            raise ValueError(
                "{} is not the data file of images.".format(image_file_name)
            )
        img_row, img_col = struct.unpack(">ii", fp.read(8))
        fig_size = img_row * img_col
        images = pyarray("B", fp.read())
    with open(os.path.join(dir_path, label_file_name), "rb") as fp:
        magic_number, record_num = struct.unpack(">ii", fp.read(8))
        if magic_number != 2049:
            raise ValueError(
                "{} is not the data file of labels.".format(label_file_name)
            )
        labels = pyarray("B", fp.read())
    digits = np.arange(10)
    index_list = [i for i in range(record_num) if labels[i] in digits]
    legal_record_num = len(index_list)
    image_data = np.zeros((legal_record_num, fig_size), dtype=np.uint8)
    label_data = np.zeros((legal_record_num, 1), dtype=np.int8)
    for i in range(legal_record_num):
        image_data[i] = np.array(
            images[index_list[i] * fig_size : (index_list[i] + 1) * fig_size]
        ).reshape(1, -1)
        label_data[i] = labels[index_list[i]]
    return image_data, label_data


def view_image(img_data: np.ndarray, img_label: np.ndarray, display_col: int) -> None:
    """按照指定列数display_col显示图像及其标签"""
    display_row = (img_label.shape[0] + display_col - 1) // display_col
    for index, (dat, lab) in enumerate(list(zip(img_data, img_label))):
        img = dat.reshape(28, 28)
        # print(img)
        plt.subplot(display_row, display_col, index + 1)
        plt.axis("off")
        plt.title("{}".format(lab), fontsize=8)
        plt.imshow(img, cmap="gray", interpolation="nearest")
    plt.show()


if __name__ == "__main__":
    dir_path = (
        "./Chapter3_MachineLearningBasic/Lab9_LogisticRegressionRecognizingNumber/data"
    )
    image_file_suffix, label_file_suffix = "-images.idx3-ubyte", "-labels.idx1-ubyte"
    # 读取数据
    train_images, train_labels = load_mnist(
        "train" + image_file_suffix, "train" + label_file_suffix, dir_path
    )
    test_images, test_labels = load_mnist(
        "test" + image_file_suffix, "test" + label_file_suffix, dir_path
    )
    # view_image(train_images[:50], train_labels[:50], 10)
    # 归一化
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # 模训练型
    model = LogisticRegression()
    model.fit(train_images, train_labels.ravel())
    # 模型评估
    test_labels_pred = model.predict(test_images)
    print(
        "accuracy score: {:.4f}".format(accuracy_score(test_labels_pred, test_labels))
    )
    print(
        "Classification report for classifier {}:\n{}\n".format(
            model, classification_report(test_labels, test_labels_pred)
        )
    )
