"""朴素贝叶斯分类新闻"""

import os, re, json, joblib
import jieba
import concurrent.futures as parallel
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# 数据标签
labels = ["其他", "体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"]
label_dict = {labels[idx]: idx for idx in range(len(labels))}
# 并发线程数
MaxThreads = 16


def read_stopwords(filepath: str) -> set:
    """读取停用词并以集合形式返回"""
    with open(filepath, "r", encoding="utf-8") as fp:
        stopwords = set([line.strip() for line in fp.readlines()])
    return stopwords


def cut_sentence(record: str, stopwords: set) -> tuple:
    """
    将给定数据分为tag和sentence,利用jieba分词sentence并过滤stopwords中的停用词形成token_list

    Return: tag, token_list
    """
    record = record.split(maxsplit=1)
    tag, sentence = record[0], record[1]
    sentence = re.sub(r"\W", "", sentence)
    token_list = [
        token
        for token in jieba.lcut(sentence)
        if (len(token) > 1) & (token not in stopwords)
    ]
    return tag, token_list


def process_origin_data(
    src_filename: str, des_filename: str, dir_path: str, stopwords: set
) -> None:
    """
    将dir_path目录下的src_filename原始数据切分提取后的结果存于同目录下的des_filename中

    Dependency: cut_sentence
    """
    with parallel.ThreadPoolExecutor(max_workers=MaxThreads) as multi_thread:
        with open(os.path.join(dir_path, src_filename), "r", encoding="utf-8") as fp:
            futures = [
                multi_thread.submit(cut_sentence, line, stopwords)
                for line in fp.readlines()
            ]
        with open(os.path.join(dir_path, des_filename), "w", encoding="utf-8") as fp:
            for future in parallel.as_completed(futures):
                tag, token_list = future.result()
                fp.write("{} {}\n".format(tag, " ".join(token_list)))


def get_token_dict(token_list_list: list) -> dict:
    """
    根据传入的token_list_list统计各token的数目以及包含该token的token_list个数;
    token_list_list嵌套的每个token_list表示一条新闻的token列表

    Return: dict(token: [tot_count, token_list_count])
    """
    token_dict = {}
    for token_list in token_list_list:
        for token in token_list:
            if token_dict.get(token) == None:
                token_dict[token] = [1, 0]
            else:
                token_dict[token][0] += 1
        for token in set(token_list):
            token_dict[token][1] += 1
    return token_dict


def read_or_build_token_dict(
    read_filename: str, build_from_filename: str, dir_path: str
) -> dict:
    """
    从dir_path目录下的read_filename以json格式加载token_dict;
    若read_filename不存在,则根据同目录下build_from_filename(extract类型的文件)
    构建token_dict并以json格式写入read_filename

    Dependency: get_token_dict
    """
    if os.path.isfile(os.path.join(dir_path, read_filename)):
        with open(os.path.join(dir_path, read_filename), "r", encoding="utf-8") as fp:
            token_dict = json.load(fp)
    else:
        with open(
            os.path.join(dir_path, build_from_filename), "r", encoding="utf-8"
        ) as fp:
            token_list_list = [line.split()[1:] for line in fp.readlines()]
            token_dict = get_token_dict(token_list_list)
        with open(os.path.join(dir_path, read_filename), "w", encoding="utf-8") as fp:
            json.dump(token_dict, fp)
    return token_dict


def get_feature_dict(
    token_dict: dict, limit: int, min_file_count: int = 1, max_file_count: int = -1
) -> dict:
    """
    选取token_dict中词频最高的至多limit个token并与其下标形成dict,
    并且选出的每个token对应的包含该token的新闻的数量不低于min_file_count不超过max_file_count
    """
    min_file_count = max(min_file_count, 1)
    max_file = max([v[1] for v in token_dict.values()])
    if (max_file_count < 0) | (max_file_count > max_file):
        max_file_count = max_file
    token_sorted = sorted(
        [
            token
            for token in token_dict.keys()
            if (token_dict[token][1] >= min_file_count)
            & (token_dict[token][1] <= max_file_count)
        ],
        key=lambda v: token_dict[v][0],
        reverse=True,
    )
    if (limit >= len(token_sorted)) | (limit < 0):
        limit = len(token_sorted)
    return {token_sorted[idx]: idx for idx in range(limit)}


def get_feature_vector(record: str, feature_dict: dict) -> tuple:
    """
    将给定数据分为tag和token_list,
    tag转换为标签的下标,
    将feature_dict中出现在token_list里的token对应位置标注1,否则置0

    Return: tag, feature_vector
    """
    record = record.split()
    tag = label_dict.get(record[0], 0)
    feature_vector = [0 for i in range(len(feature_dict))]
    for token in set(record[1:]):
        if feature_dict.get(token) != None:
            feature_vector[feature_dict[token]] = 1
    return tag, feature_vector


def process_extract_data(
    src_filename: str, des_datatype: str, dir_path: str, feature_dict: dict
):
    """
    将dir_path目录下的src_filename分词数据(extract类型文件)转为one-hot编码,
    结果存于同目录下的des_datatype对应的tags和vectors二进制文件中

    Dependency: get_feature_vector
    """
    with parallel.ThreadPoolExecutor(max_workers=MaxThreads) as multi_thread:
        with open(os.path.join(dir_path, src_filename), "r", encoding="utf-8") as fp:
            futures = [
                multi_thread.submit(get_feature_vector, line, feature_dict)
                for line in fp.readlines()
            ]
        tag_list, vector_list = [], []
        for future in parallel.as_completed(futures):
            tag, feature_vector = future.result()
            tag_list.append(tag)
            vector_list.append(feature_vector)
        save_npz(
            os.path.join(dir_path, "{}-vectors.npz".format(des_datatype)),
            csr_matrix(vector_list),
        )
        np.save(
            os.path.join(dir_path, "{}-tags.npy".format(des_datatype)),
            np.array(tag_list).reshape(-1, 1).ravel(),
        )


def read_vectors_and_tags(src_datatype: str, dir_path: str) -> tuple:
    """
    读取src_datatype对应的vector和tag二进制文件

    Return: vectors(scipy.sparse.csr_matrix), tags(numpy.ndarray)
    """
    tags = np.load(os.path.join(dir_path, "{}-tags.npy".format(src_datatype)))
    vectors = load_npz(os.path.join(dir_path, "{}-vectors.npz".format(src_datatype)))
    return vectors, tags


def load_or_fit_model(
    model_filename: str,
    dir_path: str,
    train_vectors: csr_matrix,
    train_tags: np.ndarray,
) -> MultinomialNB:
    """
    从dir_path目录下的model_filename以加载model;
    若model_filename不存在,则根据创建模型并训练保存至model_filename文件
    """
    if os.path.isfile(os.path.join(dir_path, model_filename)):
        model = joblib.load(os.path.join(dir_path, model_filename))
    else:
        # alpha设置拉普拉斯平滑, fit_prior表示是否考虑先验概率
        model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        model.fit(train_vectors, train_tags)
        joblib.dump(model, os.path.join(dir_path, model_filename))
    return model


def predict_custom_cnews(
    cnews: str, stopwords: set, feature_dict: dict, model: MultinomialNB
) -> str:
    """对用户给定的中文新闻利用给定模型预测其所属分类"""
    sentence = re.sub(r"\W", "", cnews)
    token_list = [
        token
        for token in jieba.lcut(sentence)
        if (len(token) > 1) & (token not in stopwords)
    ]
    feature_vector = [0 for i in range(len(feature_dict))]
    for token in set(token_list):
        if feature_dict.get(token) != None:
            feature_vector[feature_dict[token]] = 1
    tag_pred = model.predict(csr_matrix([feature_vector]))[0]
    tag_name = labels[tag_pred]
    return tag_name


if __name__ == "__main__":
    dir_path = "./Chapter3_MachineLearningBasic/Lab10_NaiveBayesClassifyingText/data"
    filename = "cnews-{}-{}.txt"
    # 预处理数据
    stopwords = read_stopwords(os.path.join(dir_path, "stopwords.txt"))
    # for data_type in ["train", "valid", "test"]:
    #     process_origin_data(
    #         filename.format("origin", data_type),
    #         filename.format("extract", data_type),
    #         dir_path,
    #         stopwords,
    #     )
    token_dict = read_or_build_token_dict(
        "train-token-count.json", filename.format("extract", "train"), dir_path
    )
    feature_dict = get_feature_dict(token_dict, 10000, 235, 5000)
    # for data_type in ["train", "valid", "test"]:
    #     process_extract_data(
    #         "cnews-extract-{}.txt".format(data_type),
    #         data_type,
    #         dir_path,
    #         feature_dict,
    #     )
    train_vectors, train_tags = read_vectors_and_tags("train", dir_path)
    test_vectors, test_tags = read_vectors_and_tags("test", dir_path)
    # 模型加载/训练
    model_filename = "NaiveBayes_CNews.joblib"
    model = load_or_fit_model(model_filename, dir_path, train_vectors, train_tags)
    # 模型评估
    test_accuracy = model.score(test_vectors, test_tags)
    print("mean accuracy on test set: {}".format(test_accuracy))
    test_tags_pred = model.predict(test_vectors)
    print(classification_report(test_tags, test_tags_pred))
    # 自定义测试
    custom_cnews = "黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 新浪体育讯北京时间4月27日，NBA季后赛首轮洛杉矶湖人主场迎战新奥尔良黄蜂，此前的比赛中，双方战成2-2平，因此本场比赛对于两支球队来说都非常重要，赛前双方也公布了首发阵容：湖人队：费舍尔、科比、阿泰斯特、加索尔、拜纳姆黄蜂队：保罗、贝里内利、阿里扎、兰德里、奥卡福[新浪NBA官方微博][新浪NBA湖人新闻动态微博][新浪NBA专题][黄蜂vs湖人图文直播室](新浪体育)"
    print(predict_custom_cnews(custom_cnews, stopwords, feature_dict, model))
