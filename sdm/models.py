import torch
import torchgeo
from torchgeo.datasets import RasterDataset
import matplotlib.pyplot as plt
from sdm.geo import tile_bounding_box

class Sentinel2Lidar(RasterDataset):
    is_image = True
    all_bands = [
        "s2_B1",
        "s2_B2",
        "s2_B3",
        "s2_B4",
        "s2_B5",
        "s2_B6",
        "s2_B7",
        "s2_B8",
        "s2_B8A",
        "s2_B9",
        "s2_B11",
        "s2_B12",
        "dsm",
        "dtm",
    ]
    rgb_bands = ["s2_B4", "s2_B3", "s2_B2"]

    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 10000, min=0, max=1).numpy()

        # return a plot that can be plotted in subplots
        return plt.imshow(image)

class VegHeightDataset(RasterDataset):
    is_image = False
    all_bands = ["veg_height"]

import torch
from torch.utils.data import Dataset
import rioxarray as rxr
import numpy as np
from torch.utils.data import Dataset
import rioxarray as rxr
import torch

class S2Dataset(Dataset):
    def __init__(self, file_path, input_bands, target_band, tile_size):
        self.file_path = file_path
        self.input_bands = input_bands
        self.target_band = target_band
        self.tile_size = tile_size
        self.data = rxr.open_rasterio(self.file_path, chunks=True)
        self.bbox = self.data.rio.bounds()
        self.bbox_tile_list = tile_bounding_box(*self.bbox, tile_shape=(self.tile_size, self.tile_size), resolution=self.data.rio.resolution()[0])
        self.input_data = self.data.sel(band=self.input_bands)
        self.target_data = self.data.sel(band=self.target_band).squeeze()


    def __len__(self):
        return len(self.bbox_tile_list)

    def __getitem__(self, idx):
        bbox = self.bbox_tile_list[idx]
        input_tile = self.input_data.rio.clip_box(*bbox, crs=self.data.rio.crs)
        input_tile = input_tile.where(input_tile < 3e10, np.nan)

        target_tile = self.target_data.rio.clip_box(*bbox, crs=self.data.rio.crs)


        input_tensor = torch.tensor(input_tile.values, dtype=torch.float32)
        target_tensor = torch.tensor(target_tile.values, dtype=torch.float32)

        
        # Create a mask tensor
        mask_tensor = torch.ones_like(input_tensor)
        mask_tensor[torch.isnan(input_tensor)] = 0
        mask_tensor[torch.isinf(input_tensor)] = 0
        mask_tensor[torch.isnan(target_tensor).any(dim=0)] = 0
        mask_tensor[torch.isinf(target_tensor).any(dim=0)] = 0

        # Replace NA / inf with -1
        input_tensor[torch.isnan(input_tensor)] = -1
        input_tensor[torch.isinf(input_tensor)] = -1
        target_tensor[torch.isnan(target_tensor)] = -1
        target_tensor[torch.isinf(target_tensor)] = -1

        return input_tensor, target_tensor, mask_tensor
