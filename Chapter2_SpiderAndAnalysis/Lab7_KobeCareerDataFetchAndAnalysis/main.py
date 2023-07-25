"""科比职业生涯数据的爬取与分析"""

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


def fetch_career_info(
    player_id: str, match_type: str, save_dir: str, player_name: str = ""
) -> None:
    """
    获取player_id/player_name对应的球员在match_type对应类型比赛中的数据并按照csv格式保存到save_dir路径下;
    match_type的取值有"season"常规赛,"playoff"季后赛,"allstar"全明星三种;
    player_name为空则自动替换为player_id;

    Dependency: get_html
    """
    url = "http://www.stat-nba.com/player/stat_box/{}_{}.html".format(
        player_id, match_type
    )
    response_text = get_html(url)
    if not response_text:
        print("Failed to access '{}'".format(url))
        return
    if len(player_name) == 0:
        player_name = "id" + player_id
    soup = BeautifulSoup(response_text, "lxml")
    title_ele_list = soup.select("body > div > div > table > thead > tr > th")
    stat_ele_list = soup.select("body > div > table.stat_box")
    for i in range(len(stat_ele_list)):
        stat_title = ""
        if i < len(title_ele_list):
            stat_title = title_ele_list[i].get_text().strip()
        stat_ele = stat_ele_list[i]
        stat_header_ele = stat_ele.select("thead tr th[class]")  # *[@class='stat_box']/
        stat_header = [col.get_text().strip() for col in stat_header_ele]
        stat_rows_ele = stat_ele.select("tbody tr.sort")
        stat_rows = []
        for stat_row_ele in stat_rows_ele:
            stat_cells_ele = stat_row_ele.select("td[class]")
            stat_cells = [cell.get_text().strip() for cell in stat_cells_ele]
            stat_rows.append(stat_cells)
        with open(
            os.path.join(
                save_dir, "{}_{}_{}.csv".format(player_name, match_type, stat_title)
            ),
            "w+",
            encoding="utf-8",
            newline="",
        ) as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(stat_header)
            for row in stat_rows:
                csv_writer.writerow(row)


def read_career_stat(filepath: str) -> tuple:
    """
    读取已爬取的球员比赛生涯数据

    Return: pandas.DataFrame格式的csv数据, 前者对应的表头
    """
    data = pd.read_csv(filepath, encoding="utf-8")
    col_name = data.columns.values
    return data, col_name


def view_score(
    player_name: str, match_type: str, stat_title: str, dirpath: str
) -> None:
    """
    使用matplotlib.pyplot可视化player_name对应球员在match_type类型比赛中stat_title表所统计的篮板、助攻、得分

    Dependency: read_career_stat
    """
    file_name = "{}_{}_{}.csv".format(player_name, match_type, stat_title)
    data, _ = read_career_stat(os.path.join(dirpath, file_name))
    # data = pd.DataFrame([])
    index = len(data["赛季"])
    sep = 1
    plt.figure(figsize=(10, 9))
    x = data["赛季"].values.tolist()
    xticks = list(range(1, len(x) + 1, sep))
    xlabels = [x[i - 1] for i in xticks]

    plt.rcParams["font.sans-serif"] = ["SimSun"]  # 选择宋体或其他中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 用于解决负号显示为方块的问题
    plt.suptitle("【{}】-【{}】生涯数据".format(player_name, match_type))
    ax1 = plt.subplot(3, 1, 1)
    y1 = [float(c) if c != None else -1 for c in data["篮板"].values.tolist()]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, rotation=20)
    plt.plot(range(1, len(x) + 1), y1, "ro-")
    plt.ylabel("篮板")
    ax2 = plt.subplot(3, 1, 2)
    y2 = [float(c) if c != None else -1 for c in data["助攻"].values.tolist()]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels, rotation=20)
    plt.plot(range(1, len(x) + 1), y2, "go-")
    plt.ylabel("助攻")
    ax3 = plt.subplot(3, 1, 3)
    y3 = [float(c) if c != None else -1 for c in data["得分"].values.tolist()]
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xlabels, rotation=20)
    plt.plot(range(1, len(x) + 1), y3, "bo-")
    plt.ylabel("得分")

    plt.show()


if __name__ == "__main__":
    save_dir = "./Chapter2_SpiderAndAnalysis/Lab7_KobeCareerDataFetchAndAnalysis/data"
    player_id = "195"
    player_name = "Kobe"
    match_types = ["season", "playoff", "allstar"]
    # 生涯数据爬取
    # for match_type in match_types:
    #     fetch_career_info(player_id, match_type, save_dir, player_name)
    # 生涯数据分析
    view_score(player_name, "season", "场均数据", save_dir)
    view_score(player_name, "playoff", "场均数据", save_dir)
    view_score(player_name, "allstar", "", save_dir)
    pass
