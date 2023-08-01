"""图片数据集合"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PicSet(Dataset):
    def __init__(
        self,
        feature_array: np.ndarray,
        target_array: np.ndarray,
        datatype: torch.Type = torch.float32,
        device: str = "cpu",
    ) -> None:
        self.feature_data = torch.tensor(feature_array, dtype=datatype, device=device)
        self.target_data = torch.tensor(target_array, dtype=datatype, device=device)
        if len(self.feature_data) != len(self.target_data):
            raise ValueError("number of records in features and targets does not match")

    def __len__(self) -> int:
        return len(self.feature_data)

    def __getitem__(self, index):
        return self.feature_data[index], self.target_data[index]


def get_pic_loader(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    batch_size: int = 32,
    datatype: torch.Type = torch.float32,
    device: str = "cpu",
) -> (DataLoader, DataLoader):
    """根据给定数据集得到数据加载器"""
    train_loader = DataLoader(
        PicSet(X_train, Y_train, datatype, device),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        PicSet(X_test, Y_test, datatype, device),
        batch_size=batch_size,
        shuffle=True,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    pass
