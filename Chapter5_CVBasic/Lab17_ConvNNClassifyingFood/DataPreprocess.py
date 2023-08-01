"""数据预处理"""

import os
import cv2
import numpy as np


def fill_with_zero(num: int, length: int) -> str:
    """将num转化为字符串并填充前导0直至达到length位数"""
    num_str = str(num)
    if len(num_str) > length:
        raise ValueError("length of {} is greater than {}".format(num, length))
    return "0" * (length - len(num_str)) + num_str


def rename_pics(dir_path: str) -> None:
    """将dir_path->类别名(子目录)目录下的图片按照类别名_图片序号.jpg格式重命名"""
    sub_dir_list = os.listdir(dir_path)
    for class_name in sub_dir_list:
        sub_dir_path = os.path.join(dir_path, class_name)
        pic_list = sorted(
            [
                pic_name
                for pic_name in os.listdir(sub_dir_path)
                if pic_name.endswith(".jpg")
            ]
        )
        for idx, old_pic_name in enumerate(pic_list):
            os.rename(
                os.path.join(sub_dir_path, old_pic_name),
                os.path.join(
                    sub_dir_path, "{}_{}.jpg".format(class_name, fill_with_zero(idx, 4))
                ),
            )


def get_file_path_list(cur_path: str, ext_list: list) -> list:
    """获取所有dir_path目录下后缀名属于ext_list的文件的路径列表"""
    file_path_list = []
    if os.path.isdir(cur_path):
        sub_path_list = os.listdir(cur_path)
        for sub_path in sub_path_list:
            file_path_list.extend(
                get_file_path_list(os.path.join(cur_path, sub_path), ext_list)
            )
    elif os.path.isfile(cur_path):
        if os.path.splitext(cur_path)[1] in ext_list:
            file_path_list.append(cur_path)
    else:
        raise ValueError("invalid path: '{}'".format(cur_path))
    return file_path_list


def get_origin_data_array(
    dir_path: str,
    target_names: list = None,
    pic_resize_shape: tuple = (64, 64),
) -> (np.ndarray, np.ndarray, list):
    """
    从dir_path目录下获取所有图片数据信息,\n
    要求dir_path内部文件层级满足: dir_path->类别名(子目录)->类别名_图片序号.jpg(图片),\n
    若给定target_names, 则按照其进行one-hot编码, 否则按照读取的文件的类别集合进行one-hot编码,\n
    读取的图片将按照pic_resize_shape缩放得到实际像素矩阵,\n
    读取处理过的数据将保存到dir_path目录下对应npz文件中

    Return: 所有图片的像素矩阵, 所有图片标签的one-hot矩阵, 所有分类的名称列表
    """
    processed_data_filepath = os.path.join(
        dir_path,
        "processed_data_{}x{}.npz".format(pic_resize_shape[0], pic_resize_shape[1]),
    )
    if os.path.isfile(processed_data_filepath):
        processed_data = np.load(processed_data_filepath)
        feature_array = processed_data["feature_array"]
        target_array = processed_data["target_array"]
        target_names = processed_data["target_names"].tolist()
        return feature_array, target_array, target_names
    pic_path_list = get_file_path_list(dir_path, [".jpg"])
    if len(pic_path_list) == 0:
        raise ValueError("no picture(.jpg) found under path '{}'".format(dir_path))
    feature_array, target_array = [], []
    for pic_path in pic_path_list:
        target = pic_path.split(os.path.sep)[-1].split("_")[0]
        target_array.append(target)
        feature = cv2.imread(pic_path)
        feature = cv2.resize(feature, pic_resize_shape)
        feature = cv2.cvtColor(feature, cv2.COLOR_BGR2RGB)
        feature = feature.transpose((2, 0, 1))
        feature_array.append(feature)
    feature_array = np.array(feature_array, dtype=np.float32) / 255.0
    if target_names is None:
        target_names = sorted(list(set(target_array)))
    target_array = np.array(
        [
            [0 if destag != srctag else 1 for destag in target_names]
            for srctag in target_array
        ],
        dtype=np.float32,
    )
    np.savez(
        processed_data_filepath,
        feature_array=feature_array,
        target_array=target_array,
        target_names=np.array(target_names),
    )
    return feature_array, target_array, target_names


if __name__ == "__main__":
    dir_path = r".\Chapter5_CVBasic\Lab17_ConvNNClassifyingFood\data\FoodPic-5Class"
    # rename_pics(dir_path)
    pass
