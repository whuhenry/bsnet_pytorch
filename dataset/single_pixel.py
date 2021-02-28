from osgeo import gdal
from torch.utils.data.dataset import Dataset
import torch
import numpy as np

class SinglePixelDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        dataset = gdal.Open(str(path), gdal.GA_ReadOnly) # type: gdal.Dataset
        self.rows = dataset.RasterYSize
        self.cols = dataset.RasterXSize
        self.bands = dataset.RasterCount

        self.data_array = dataset.ReadAsArray()
        self.data_array = self.data_array.astype(np.float32)
        self.min_val = np.min(self.data_array)
        self.max_val = np.max(self.data_array)
        self.data_array = (self.data_array - self.min_val) / (self.max_val - self.min_val)
        del dataset
        dataset = None

    def __len__(self):
        return self.rows * self.cols

    def __getitem__(self, idx):
        y = idx // self.cols
        x = idx - y * self.cols
        return torch.FloatTensor(self.data_array[:, y, x])