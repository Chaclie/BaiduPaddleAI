"""电影海报的爬取"""

import os
import json
import time
import requests
import lxml
from urllib.parse import quote
from bs4 import BeautifulSoup


WaitSeconds = 2


def get_response(url: str, referer: str) -> requests.Response:
    """通过指定url获取页面响应"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.0.0",
        "Accept": "*/*",
        "Referer": referer,
        "Connection": "keep-alive",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "Accept-Encoding": "gzip, deflate, br, zsdch",
    }
    response = requests.get(url, headers)
    time.sleep(WaitSeconds)
    if response.status_code == 200:
        return response
    else:
        return None


def fetch_image(search_keyword: str, save_dir: str) -> None:
    """
    利用bing搜索search_keyword并将查找的的部分图片存储在save_dir目录下

    Dependency: get_response
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    search_url = "https://www.bing.com/images/search?q={}&first=1".format(
        # 用于将非安全字符转换为url中的安全编码
        quote(search_keyword)
    )
    response = get_response(search_url, "")
    if response is not None:
        soup = BeautifulSoup(response.text, "lxml")
        images_list = soup.select("ul.dgControl_list")
        col_max, img_max = 3, 7
        col_id = 0
        for image_list in images_list:
            col_id += 1
            if col_id > col_max:
                break
            image_element_list = image_list.select("li img")
            img_id = 0
            for image_element in image_element_list:
                img_id += 1
                if img_id > img_max:
                    break
                image_url = image_element.get("src")
                if image_url is None:
                    continue
                image_response = get_response(image_url, search_url)
                if image_response is not None:
                    ext = image_response.headers["Content-Type"].split("/")[-1]
                    with open(
                        os.path.join(save_dir, "{}-{}.{}".format(col_id, img_id, ext)),
                        "wb",
                    ) as fp:
                        fp.write(image_response.content)
                else:
                    print("Failed to access '{}'".format(image_url))
    else:
        print("Failed to access '{}'".format(search_url))


if __name__ == "__main__":
    fetch_image(
        "电影海报", "./Chapter2_SpiderAndAnalysis/Lab5_StarPictureFetch/data/images"
    )
    pass
