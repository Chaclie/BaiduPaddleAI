import os, cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def get_file_path_list(cur_path: str, ext_list: list = []) -> list:
    """获取cur_path下所有扩展名属于ext_list指定范围的文件的路径列表, 若有嵌套文件夹则递归获取"""
    file_path_list = []
    if os.path.isdir(cur_path):
        nxt_path_list = os.listdir(cur_path)
        for nxt_path in nxt_path_list:
            file_path_list.extend(
                get_file_path_list(os.path.join(cur_path, nxt_path), ext_list)
            )
    elif os.path.isfile(cur_path):
        if os.path.splitext(cur_path)[1] in ext_list:
            file_path_list.append(cur_path)
    return file_path_list


def get_origin_data_array(
    dir_path: str, pic_resize_shape: tuple, targets: list = None
) -> (np.ndarray, np.ndarray, list):
    """
    获取dir_path目录下的所有文件并按照jpg格式读取,\n
    按照pic_resize_shape指定的大小重塑图片形状,\n
    将所有标记按照one_hot编码

    Dependency: get_file_path_list

    Return: 所有条目的特征, 所有条目的one_hot标记, 所有原始标记名称列表
    """
    pic_path_list = get_file_path_list(dir_path, [".jpg"])
    if not pic_path_list:
        raise ValueError(
            "no picture(.jpg) detected under directory '{}'".format(dir_path)
        )
    feature_origin_data, target_origin_data = [], []
    for pic_path in pic_path_list:
        feature = cv2.imread(pic_path)
        feature = cv2.resize(feature, pic_resize_shape)
        target = pic_path.split(os.path.sep)[-1].split("_")[0]
        feature_origin_data.append(feature)
        target_origin_data.append(target)
    feature_origin_data = np.array(feature_origin_data)
    if targets is None:
        targets = sorted([target for target in set(target_origin_data)])
    target_origin_data = np.array(
        [
            [0 if tardes != tarsrc else 1 for tardes in targets]
            for tarsrc in target_origin_data
        ]
    )
    return feature_origin_data, target_origin_data, targets


class CustomDataset(Dataset):
    """自定义宝石数据集"""

    def __init__(
        self,
        feature_origin_data: np.ndarray,
        target_origin_data: np.ndarray,
        datatype=torch.float32,
        device: str = "cpu",
    ) -> None:
        """
        利用给定的origin_data初始化特征和标记,\n
        存储数据类型为datatype,\n
        数据存储的设备通过device指定
        """
        self.feature_data = torch.tensor(
            feature_origin_data, dtype=datatype, device=device
        )
        self.feature_data /= 255.0
        self.target_data = torch.tensor(
            target_origin_data, dtype=datatype, device=device
        )

    def __len__(self) -> int:
        """获取数据集条目数"""
        return len(self.feature_data)

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor):
        """获取index对应编号的特征和标记张量"""
        return self.feature_data[index], self.target_data[index]


if __name__ == "__main__":
    pass
