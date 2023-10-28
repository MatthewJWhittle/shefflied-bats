import matplotlib.pyplot as plt
from sdm.geo import tile_bounding_box
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import rioxarray as rxr
import torch.nn.functional as F
from torchvision.transforms import v2

# import vision tensors
from torchvision.tv_tensors import Image, Mask
import xarray as xr


class S2Dataset(Dataset):
    def __init__(self, transforms=None, image_size=(304, 304)):
        # File Path
        self.file_path = "data/processed/s2-lidar/s2-lidar-stack.tif"
        # Load the data lazily
        self.data = rxr.open_rasterio(self.file_path, chunks=True)

        # Transforms
        # These are applied to the input and target tensors to augment the data
        self.transforms = transforms

        # Band Names and Indexes
        self.input_bands = [1, 2, 3, 4, 5, 6, 7]
        self.target_band = self.input_bands[-1] + 1

        # Grab the relevant bands
        self.input_data = self.data.sel(band=self.input_bands)  # type: ignore
        self.target_data = self.data.sel(band=self.target_band).squeeze()  # type: ignore

        # Load the bounding boxes which are used to generate the img tiles
        self.bbox = self.data.rio.bounds()
        self.crs = self.data.rio.crs
        self.resolution = self.data.rio.resolution()

        # Load the bounding boxes
        self._load_bboxs()
        self.image_size = image_size

        # Set the normalisation values for each band
        band_norm_values = [10_000, 10_000, 10_000, 10_000, 10_000, 10_000, 30]
        assert len(band_norm_values) == len(
            self.input_bands
        ), "Band norm values must be the same length as the number of input bands"
        self.band_norm_values = torch.tensor(band_norm_values, dtype=torch.float32)

        # Initialise spatial info dict
        self.spatial_info = {}

    def _load_bboxs(self):
        """Load the bounding box csv. These are all the bboxes that don't contain NA values."""
        bbox_path = "data/processed/s2-lidar/valid-bounding-boxes.csv"
        bbox_df = pd.read_csv(bbox_path)

        self.bbox_tile_list = []
        for i, row in bbox_df.iterrows():
            bbox = (row.xmin, row.ymin, row.xmax, row.ymax)
            self.bbox_tile_list.append(bbox)

    def __len__(self):
        return len(self.bbox_tile_list)

    def __getitem__(self, idx):
        bbox = self.bbox_tile_list[idx]
        input_tile = self.input_data.rio.clip_box(*bbox, crs=self.data.rio.crs)
        target_tile = self.target_data.rio.clip_box(*bbox, crs=self.data.rio.crs)

        # Replace NA values with 0
        input_tile = input_tile.fillna(0)
        target_tile = target_tile.fillna(0)

        input_tensor = torch.tensor(input_tile.values, dtype=torch.float32)
        target_tensor = torch.tensor(target_tile.values, dtype=torch.float32)

        # Normalize the input tensor
        input_tensor = input_tensor / self.band_norm_values[:, None, None]

        # Pad the tensors to the same size
        input_tensor, input_padding = self._pad_tensor(input_tensor, self.image_size)
        target_tensor, target_padding = self._pad_tensor(target_tensor, self.image_size)

        # Collect spatial information
        self._collect_spatial_info(
            idx,
            bbox=bbox,
            rio_transform=input_tile.rio.transform(),
            coords=input_tile.coords,
            padding=input_padding,
        )

        # Convert to image and mask tensors
        input_tensor = Image(input_tensor)
        target_tensor = Mask(target_tensor)

        # Apply transforms to the tensors to augment the data
        if self.transforms:
            input_tensor, target_tensor = self.transforms(input_tensor, target_tensor)

        return input_tensor, target_tensor, torch.LongTensor([idx])

    def _collect_spatial_info(
        self, idx, bbox, rio_transform, coords, padding: tuple
    ) -> None:
        """Collect the spatial information needed to convert the model output back to a raster."""

        self.spatial_info[idx] = {
            "bbox": bbox,
            "transform": rio_transform,
            "padding": padding,
            "crs": self.crs,
            "coords": coords,
        }

    def _pad_tensor(self, tensor, target_shape):
        """Pad tensor to the target shape."""
        # Calculate padding amounts
        pad_height = target_shape[0] - tensor.shape[-2]
        pad_width = target_shape[1] - tensor.shape[-1]

        # Pad tensor. Format: (left, right, top, bottom)
        padded_tensor = F.pad(tensor, (0, pad_width, 0, pad_height), "constant", 0)

        return padded_tensor, (pad_height, pad_width)

from typing import Union

class SpatialTransformer:
    """Transforms the model output back to a raster."""

    def __init__(self, bbox, transform, padding, crs, coords):
        self.bbox = bbox
        self.transform = transform
        self.crs = crs
        self.coords = coords
        # H then W
        self.padding = padding

    def __call__(self, tensor:Union[torch.Tensor, np.ndarray], bands=1) -> xr.DataArray:
        """
        Convert the model output tensor to an xarray DataArray.

        Args:
            tensor (torch.Tensor): The model output tensor.
        Returns:
            xarray.DataArray: The model output as an xarray DataArray.
                
        """
        # Conver the tensor to a numpy array
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        
        # Remove the padding fro the array
        if bands == 1:
            tensor = tensor[: tensor.shape[-2] - self.padding[0], : tensor.shape[-1] - self.padding[1]]
        else:
            tensor = tensor[:, : tensor.shape[-2] - self.padding[0], : tensor.shape[-1] - self.padding[1]]
        
        # Check coords match shape of tensor 
        assert tensor.shape[-2] == len(self.coords["y"])
        assert tensor.shape[-1] == len(self.coords["x"])

        # Generate the xarray data array
        da = xr.DataArray(
            tensor,
            dims=["y", "x"],
            coords={"x": self.coords["x"], "y": self.coords["y"]},
            attrs={"crs": self.crs, "transform": self.transform},
        )

        da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        da.rio.write_crs(self.crs, inplace=True)

        return da


def plot_predictions(target, prediction):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(target, cmap='viridis')
    ax1.set_title('Mask')
    ax2.imshow(prediction, cmap='viridis')
    ax2.set_title('Prediction')
    plt.show()
