"""图像数据基本预处理"""

import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def show_image(image: np.ndarray, title: str = None, cmap: str = None) -> None:
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.imshow(image, cmap=cmap)


if __name__ == "__main__":
    dir_path = r"./Chapter5_CVBasic/Lab16_PicDataPreprocess/data"
    gray_pic_name, rgb_pic_name = "Lena-gray.jpg", "Lena-rgb.png"
    gray_pic_path = os.path.join(dir_path, gray_pic_name)
    rgb_pic_path = os.path.join(dir_path, rgb_pic_name)
    # Image读取&显示
    img_gray = Image.open(gray_pic_path)
    img_rgb = Image.open(rgb_pic_path)
    # 以下两种获取通道图像值等价
    img_r_ch, img_g_ch, img_b_ch = img_rgb.split()
    img_r_ch = img_rgb.getchannel(0)
    img_g_ch = img_rgb.getchannel(1)
    img_b_ch = img_rgb.getchannel(2)
    plt.subplot(2, 2, 1), show_image(img_rgb)
    plt.subplot(2, 2, 2), show_image(img_r_ch, cmap="Reds")
    plt.subplot(2, 2, 3), show_image(img_g_ch, cmap="Blues")
    plt.subplot(2, 2, 4), show_image(img_b_ch, cmap="Greens")
    plt.suptitle("PIL.Image")
    # plt.show()
    # cv2读取&显示
    img_gray = cv2.imread(gray_pic_path)
    img_bgr = cv2.imread(rgb_pic_path)
    height, width, channels = img_bgr.shape
    img_b_ch, img_g_ch, img_r_ch = cv2.split(img_bgr)
    img_merge_ch = cv2.merge([img_b_ch, img_g_ch, img_r_ch])
    img_rgb = cv2.cvtColor(img_merge_ch, cv2.COLOR_BGR2RGB)
    plt.clf()
    plt.subplot(2, 2, 1), show_image(img_rgb)
    plt.subplot(2, 2, 2), show_image(img_r_ch, cmap="Reds")
    plt.subplot(2, 2, 3), show_image(img_g_ch, cmap="Blues")
    plt.subplot(2, 2, 4), show_image(img_b_ch, cmap="Greens")
    plt.suptitle("cv2")
    # plt.show()
    # cv2.imshow("cv2-bgr", img_bgr)  # 非阻塞显示
    # 图像切分&拼接
    # 使用numpy数组的切片和拼接即可
    # 图像缩放
    img_bgr_resize1 = cv2.resize(
        src=img_bgr, dsize=(1024, 256), interpolation=cv2.INTER_CUBIC
    )
    # cv2.imshow("cv2-bgr-128*256", img_bgr_resize1)
    img_rgb_resize2 = cv2.resize(src=img_rgb, dsize=None, fx=2.0, fy=1.0)
    img_rgb_resize3 = cv2.resize(src=img_rgb, dsize=None, fx=1.0, fy=2.0)
    plt.clf()
    plt.subplot(1, 2, 1), show_image(img_rgb_resize2)
    plt.subplot(1, 2, 2), show_image(img_rgb_resize3)
    # plt.show()
    # 二值化处理
    ret1, img_rgb_thr1 = cv2.threshold(
        img_rgb, 100, 200, cv2.THRESH_BINARY
    )  # 超过阈值则设置为指定的maxval,否则置0; 返回的ret为输入的阈值
    ret2, img_rgb_thr2 = cv2.threshold(
        img_rgb, 100, 200, cv2.THRESH_BINARY_INV
    )  # 超过阈值则设置为0,否则指定的maxval
    ret3, img_rgb_thr3 = cv2.threshold(
        img_rgb, 100, 200, cv2.THRESH_TRUNC
    )  # 超过阈值则设置为指定的maxval,否则不变
    ret4, img_rgb_thr4 = cv2.threshold(
        img_rgb, 100, 200, cv2.THRESH_TOZERO
    )  # 超过阈值则不变,否则置0
    ret5, img_rgb_thr5 = cv2.threshold(
        img_rgb, 100, 200, cv2.THRESH_TOZERO_INV
    )  # 超过阈值则置0,否则不变
    plt.clf()
    plt.subplot(231), show_image(img_rgb_thr1)
    plt.subplot(232), show_image(img_rgb_thr2)
    plt.subplot(233), show_image(img_rgb_thr3)
    plt.subplot(234), show_image(img_rgb_thr4)
    plt.subplot(235), show_image(img_rgb_thr5)
    # plt.show()
    # 图像归一化
    # 对对应的numpy数组或torch张量归一化即可
