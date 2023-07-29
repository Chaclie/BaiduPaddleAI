"""自定义数据集"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    自定义数据集: 继承自torch.utils.data.Dataset
    """

    def __init__(
        self,
        feature_matrix: np.matrix = None,
        target_matrix: np.matrix = None,
        feature_file: str = None,
        target_file: str = None,
        feature_transform=None,
        target_transform=None,
        datatype=torch.float32,
        device=None,
    ) -> None:
        """
        feature表示特征, target表示标记,\n
        matrix指明数据存储的矩阵, file指明数据源路径(需保证为csv格式), 二者指定其一即可,\n
        transform指明数据变换形式\n
        device指明数据转移的设备
        """
        if feature_matrix is not None:
            self.feature_data = feature_matrix
        elif feature_file is not None:
            self.feature_data = pd.read_csv(feature_file).values
        else:
            raise ValueError("features of dataset are not specified")
        if target_matrix is not None:
            self.target_data = target_matrix
        elif target_file is not None:
            self.target_data = pd.read_csv(target_file).values
        else:
            raise ValueError("targets of dataset are not specified")
        if self.feature_data.shape[0] != self.target_data.shape[0]:
            raise ValueError("number of records in features and targets does not match")
        self.feature_data = torch.tensor(self.feature_data, dtype=datatype)
        self.target_data = torch.tensor(self.target_data, dtype=datatype)
        if device is not None:
            self.feature_data = self.feature_data.to(device=device)
            self.target_data = self.target_data.to(device=device)
        self.feature_transform = feature_transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """获取数据集条目数"""
        return self.feature_data.shape[0]

    def __getitem__(self, index):
        """
        返回对应index的数据的特征和标记,
        如已设定transform方式则执行转换后再返回
        """
        record_feature = self.feature_data[index, :]
        if self.feature_transform:
            record_feature = self.feature_transform(record_feature)
        record_target = self.target_data[index, :]
        if self.target_transform:
            record_target = self.target_transform(record_target)
        return record_feature, record_target


if __name__ == "__main__":
    pass
