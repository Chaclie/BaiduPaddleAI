"""股票数据的爬取与分析"""

import os
import csv
import json
import time
import random
import requests
import lxml
import threading
import urllib.request
from urllib.parse import quote
from bs4 import BeautifulSoup

import pandas as pd
import matplotlib.pyplot as plt


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


def get_html(url: str) -> str:
    """获取指定url的页面内容"""
    response = requests.get(url, headers={"User-Agent": random.choice(UserAgent_list)})
    time.sleep(WaitSeconds)
    if response.status_code == 200:
        response.encoding = response.apparent_encoding
        return response.text
    else:
        return ""


def fetch_stock_info(url: str, save_path: str, limit: int = 10) -> None:
    """
    获取指定url的不超过limit条股票数据并按照csv格式保存到save_path路径下

    Dependency: get_html
    """
    response_text = get_html(url)
    if not response_text:
        print("Failed to access '{}'".format(url))
        return
    json_text = response_text.split("(")[1].split(")")[0]
    result_json = json.loads(json_text)
    data_list = result_json["data"]["diff"]
    data_extract_list = []
    for data in data_list:
        data_extract_list.append([data["f12"], data["f14"]])
        if len(data_extract_list) == limit:
            break
    with open(save_path, "w+", encoding="utf-8", newline="") as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(("代码", "名称"))
        for data in data_extract_list:
            csv_writer.writerow((data[0], data[1]))


def get_stock_list(filepath: str) -> list:
    """读取已爬取的股票数据"""
    stock_list = []
    with open(filepath, "r", encoding="utf-8") as fp:
        csv_reader = csv.reader(fp)
        # 跳过表头
        next(csv_reader)
        for record in csv_reader:
            stock_list.append(record)
    return stock_list


def fetch_stock_kline(
    save_dir: str, stock_name: str, stock_code: str, start_date: str, end_date: str
) -> None:
    """
    将stock_code/start_date/end_date指定相关查询页面结果通过json
    提取日k数据保存到save_dir目录下的stock_name名字的csv中;
    stock_code以是否为3开头分别表示沪市0和深市1;
    start_date/end_date以yyyymmdd的格式给出

    Dependency: get_html
    """
    url = (
        "https://push2his.eastmoney.com/api/qt/stock/kline/get?cb=jQuery35100370690280741266_1690211246034"
        + "&secid={}.{}&ut=fa5fd1943c7b386f172d6893dbfba10b"
        + "&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61"
        + "&klt=101&fqt=1&beg={}&end={}&smplmt=460&lmt=1000000&_=1690211246075"
    ).format(0 if stock_code.startswith("3") else 1, stock_code, start_date, end_date)
    response_text = get_html(url)
    if not response_text:
        print("Failed to access kline info of '{} {}'".format(stock_code, stock_name))
        return
    json_text = response_text.split("(")[1].split(")")[0]
    result_json = json.loads(json_text)
    days_list = result_json["data"]["klines"]
    data_extract_list = []
    for day_list in days_list:
        data_list = day_list.split(",")
        data_extract_list.append(data_list)
    with open(
        os.path.join(save_dir, stock_name + ".csv"), "w+", encoding="utf-8", newline=""
    ) as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(
            ("日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率")
        )
        for data in data_extract_list:
            csv_writer.writerow(data)


def download_multithread(stock_list: list, save_dir: str) -> None:
    """
    通过信号量多线程获取股票日k信息(从2020.01.01开始)

    Dependency: fetch_stock_kline
    """
    semaphore = threading.Semaphore(1)

    def download_semaphore(stock_name: str, stock_code: str) -> None:
        with semaphore:
            fetch_stock_kline(save_dir, stock_name, stock_code, "20200101", "20500101")

    thread_list = []
    for stock in stock_list:
        tmp_thread = threading.Thread(
            target=download_semaphore, args=(stock[1], stock[0])
        )
        tmp_thread.start()
        thread_list.append(tmp_thread)

    for thrd in thread_list:
        thrd.join()


def read_stock_kline(filepath: str) -> tuple:
    """
    读取已爬取的股票日k信息

    Return: pandas.DataFrame格式的csv数据, 前者对应的表头
    """
    data = pd.read_csv(filepath, encoding="utf-8")
    col_name = data.columns.values
    return data, col_name


def view_stock_diff(stock_name: str, dirpath: str) -> None:
    """
    使用matplotlib.pyplot可视化对应股票涨跌额/幅数据

    Dependency: read_stock_kline
    """
    data, _ = read_stock_kline(os.path.join(dirpath, stock_name + ".csv"))
    # data = pd.DataFrame([])
    index = len(data["日期"]) - 1
    sep = index // 15
    plt.figure(figsize=(15, 17))
    x = data["日期"].values.tolist()
    xticks = list(range(0, len(x), sep))
    xlabels = [x[i] for i in xticks]

    plt.rcParams["font.sans-serif"] = ["SimSun"]  # 选择宋体或其他中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 用于解决负号显示为方块的问题
    plt.suptitle("【{}】-涨跌额/涨跌幅".format(stock_name))
    y1 = [float(c) if c != None else 0 for c in data["涨跌额"].values.tolist()]
    y2 = [float(c) if c != None else 0 for c in data["涨跌幅"].values.tolist()]
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, rotation=20)
    plt.plot(range(1, len(x) + 1), y1, c="r", linestyle="-")
    plt.plot(range(1, len(x) + 1), y2, c="b", linestyle="--")
    plt.legend(["涨跌额", "涨跌幅"])
    plt.show()


def view_stock_deal_diff_relation(stock_name: str, dirpath: str) -> None:
    """
    使用matplotlib.pyplot可视化对应股票成交量与前一日涨跌额的散点关系

    Dependency: read_stock_kline
    """
    data, _ = read_stock_kline(os.path.join(dirpath, stock_name + ".csv"))

    plt.rcParams["font.sans-serif"] = ["SimSun"]  # 选择宋体或其他中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 用于解决负号显示为方块的问题
    plt.title("【{}】-成交量&前一日涨跌额".format(stock_name))
    x = [float(c) if c != None else 0 for c in data["涨跌额"].values.tolist()]
    plt.xlabel("前一日涨跌额")
    y = [float(c) if c != None else 0 for c in data["成交量"].values.tolist()]
    y = [0] + y[:-1]
    plt.ylabel("成交量")
    plt.scatter(x, y, alpha=0.4)
    plt.show()


if __name__ == "__main__":
    stock_url = "http://57.push2.eastmoney.com/api/qt/clist/get?cb=jQuery112405162984108051485_1690196297321&pn=1&pz=20&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=|0|0|0|web&fid=f3&fs=m:0+t:80&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152&_=1690196297600"
    save_dir = "./Chapter2_SpiderAndAnalysis/Lab6_StockDataFetchAndAnalysis/data"
    # 股票数据爬取
    # filepath_stock_csv = os.path.join(save_dir, "stock.csv")
    # fetch_stock_info(stock_url, filepath_stock_csv, 20)
    # download_multithread(get_stock_list(os.path.join(save_dir, "stock.csv")), save_dir)
    # 股票数据分析
    # view_stock_diff("指南针", save_dir)
    view_stock_deal_diff_relation("山水比德", save_dir)
    pass
