"""实现文本词频的统计分析"""

import os
import jieba


def read_stopwords(filepath: str) -> list:
    """读取停用词表"""
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as fp:
            stopwords = fp.readlines()
        return [line.strip() for line in stopwords]
    else:
        return []


def get_token_distribution(filepath: str, stopwords: list, limit: int = 10) -> list:
    """统计中文文本词频"""
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as fp:
            text = fp.read()
        token_list = jieba.lcut(text)
        token_dict = {}
        for token in token_list:
            if token not in stopwords:
                if len(token) > 1:
                    token_dict[token] = token_dict.get(token, 0) + 1
        token_sorted_distribution = sorted(
            list(token_dict.items()), key=lambda ele: ele[1], reverse=True
        )
        if (0 < limit) & (limit < len(token_sorted_distribution)):
            return token_sorted_distribution[:limit]
        else:
            return token_sorted_distribution
    else:
        raise ValueError("给定路径'{}'不是一个文件".format(filepath))


if __name__ == "__main__":
    filepath_stopwords = (
        "./Chapter1_PythonBasic/Lab4_WordFrequencyAnalysis/data/Stopwords_SCU.txt"
    )
    filepath_text = "./Chapter1_PythonBasic/Lab4_WordFrequencyAnalysis/data/Father.txt"
    stopwords = read_stopwords(filepath_stopwords)
    print(get_token_distribution(filepath_text, stopwords))
    pass
