"""数据集加载"""
import os
import csv
import time
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR


# 每爬取请求一次页面后挂起时长
WaitSeconds = 2
# User-Agent随机列表
UserAgent_list = [
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.0.0",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
]


def get_page_text(url: str) -> str:
    """获取指定url的页面内容"""
    response = requests.get(url, headers={"User-Agent": random.choice(UserAgent_list)})
    time.sleep(WaitSeconds)
    if response.status_code == 200:
        return response.text
    else:
        return ""


def fetch_housing_csv(save_dir: str) -> None:
    """
    将Boston房价数据按照csv格式写入save_dir目录下的housing.csv中

    Dependency: get_page_text
    """
    page_text = get_page_text(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    )
    page_text = page_text.strip()
    row_list = page_text.split("\n")
    data_extracted = []
    for row in row_list:
        data_extracted.append(row.strip().split())
    with open(
        os.path.join(save_dir, "housing.csv"), "w", encoding="utf-8", newline=""
    ) as fp:
        writer = csv.writer(fp)
        header = [
            # 城镇人均犯罪率
            "CRIM",
            # 占地超25000平方英尺的住宅用地占比
            "ZN",
            # 城镇非商业用地占比
            "INDUS",
            # 是否毗邻河道
            "CHAS",
            # 一氧化氮浓度
            "NOX",
            # 每栋住宅平均房间数
            "RM",
            # 1940前建造的自住单位比例
            "AGE",
            # 与波士顿5个就业中心加权距离
            "DIS",
            # 距离高速公路的便利指数
            "RAD",
            # 每万美元不动产税率
            "TAX",
            # 城镇生师比
            "PTRATIO",
            # 黑人比例
            "B",
            # 房东属于低收入阶层比例
            "LSTAT",
            # 自住房屋房价中位数(单位:千美元)
            "MEDV",
        ]
        writer.writerow(header)
        writer.writerows(data_extracted)


def normalize_min_max_avg(data: pd.Series) -> pd.Series:
    """利用最值和均值归一化数据"""
    min_val = data.min()
    max_val = data.max()
    avg_val = data.mean()
    if max_val == min_val:
        if not max_val:
            return data
        else:
            return data / max_val
    else:
        return (data - avg_val) / (max_val - min_val)


def divide_train_test(
    data: pd.DataFrame, train_set_ratio: float, rand_seed: int
) -> tuple:
    """
    按照seed指定的随机序列重排data,并将前ratio占比条作为train_set,其后作为test_set

    Return: train_set, test_set
    """
    shuffled_order = [i for i in range(len(data))]
    random.seed(rand_seed)
    random.shuffle(shuffled_order)
    train_test_divide = int(len(housing_df_norm) * train_set_ratio)
    train_set = data.iloc[shuffled_order[:train_test_divide]]
    test_set = data.iloc[shuffled_order[train_test_divide:]]
    return train_set, test_set


def split_x_y(data: pd.DataFrame, y_col: list, x_col: list = None) -> tuple:
    """
    按照x_col和y_col指定的列名拆分data的特征和标签,x_col为None时默认将除了y_col指定的列之外所有列作为特征列

    Return: x_df, y_df
    """
    header = data.columns.tolist()
    y_col = [col for col in y_col if col in header]
    if x_col == None:
        x_col = [col for col in header if col not in y_col]
    x_df = data[x_col]
    y_df = data[y_col]
    return x_df, y_df


def get_linear_regression_model() -> LR:
    """获取一个线性回归模型"""
    return LR()


def train_model(model: LR, train_x: pd.DataFrame, train_y: pd.DataFrame) -> None:
    """根据训练集特征和标签训练线性回归模型"""
    train_x_matrix = train_x.to_numpy()
    train_y_matrix = train_y.to_numpy()
    model.fit(train_x_matrix, train_y_matrix)


def view_infer_result(
    ground_truths: list | np.ndarray,
    infer_results: list | np.ndarray,
    fig_title: str = "",
) -> None:
    """根据真实值和预测值绘制评估图，fig_title指明图表标题"""
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
    if fig_title:
        plt.title(fig_title)
    plt.plot([left_end, right_end], [left_end, right_end], "g-")
    plt.scatter(ground_truths, infer_results, marker="x", c="r")
    plt.show()


if __name__ == "__main__":
    # 数据集加载
    save_dir = (
        "./Chapter3_MachineLearningBasic/Lab8_LinearRegressionPredictingHousePrice/data"
    )
    # fetch_housing_csv(save_dir)
    housing_df = pd.read_csv(os.path.join(save_dir, "housing.csv"))
    housing_header = housing_df.columns.tolist()
    housing_df_norm = housing_df
    # 归一化
    for col_name in housing_header:
        housing_df_norm[col_name] = normalize_min_max_avg(housing_df_norm[col_name])
    # 训练-测试集划分
    train_house_df, test_house_df = divide_train_test(housing_df_norm, 0.8, 52)
    y_col_list = ["MEDV"]
    train_x_df, train_y_df = split_x_y(train_house_df, y_col_list)
    test_x_df, test_y_df = split_x_y(test_house_df, y_col_list)
    # 模型训练
    lr_model = get_linear_regression_model()
    train_model(lr_model, train_x_df, train_y_df)
    # 模型预测
    test_y_pred = lr_model.predict(test_x_df.to_numpy())
    # 模型评估
    view_infer_result(test_y_df.values, test_y_pred, "Boston房价")
