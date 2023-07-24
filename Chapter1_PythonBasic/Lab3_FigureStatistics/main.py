"""实现图像信息的直方图统计"""

import os
import cv2
from matplotlib import pyplot as plt


def show_RGB_sum_distribution(filepath_image: str) -> None:
    """统计RGB三通道的灰度值分布的累加"""
    if os.path.isfile(filepath_image):
        img = cv2.imread(filepath_image)
        plt.hist(img.reshape([-1]), 256, [0, 256])
        plt.show()
    else:
        raise ValueError("给定路径'{}'不是一个文件".format(filepath_image))


def show_RGB_each_distribution(filepath_image: str) -> None:
    """统计RGB三通道各自的灰度值分布"""
    if os.path.isfile(filepath_image):
        img = cv2.imread(filepath_image)
        color = ("r", "g", "b")
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()
    else:
        raise ValueError("给定路径'{}'不是一个文件".format(filepath_image))


if __name__ == "__main__":
    filepath_spongebob_img = (
        "./Chapter1_PythonBasic/Lab3_FigureStatistics/data/images/SpongeBob.png"
    )
    show_RGB_sum_distribution(filepath_spongebob_img)
    show_RGB_each_distribution(filepath_spongebob_img)
    pass
