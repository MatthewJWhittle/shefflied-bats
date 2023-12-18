from pathlib import Path
import xarray as xr
import rioxarray as rxr


def load_dataset(path: Path):
    data = rxr.open_rasterio(path)
    # Extract the band name
    long_name = data.attrs["long_name"]
    if type(long_name) == str:
        long_name = [long_name]
    else:
        long_name = list(long_name)

    # prefix the band name with the file name
    long_name = [f"{path.stem}_{name}" for name in long_name]

    # Rename the band dimension and convert to a dataset
    data.coords["band"] = long_name
    return data.to_dataset(dim="band")


def load_evs(ev_folder: Path):
    # list the tifs
    ev_tifs = list(ev_folder.glob("*.tif"))

    evs = [load_dataset(path) for path in ev_tifs]

    evs = xr.merge(evs)

    return evs


def interpolate_nas(dataset: xr.Dataset) -> xr.Dataset:
    dataset = dataset.sortby("y")
    dataset = dataset.interpolate_na(dim="y")
    dataset = dataset.interpolate_na(dim="x")

    return dataset


def calculate_multiscale_variables(dataset: xr.Dataset, window: int) -> xr.Dataset:
    vars = (
        dataset.rolling(x=window, y=window, center=True)
        .mean(skipna=True)
        .rename(
            {name: f"{name}_{round((window/2) * 100)}m" for name in dataset.data_vars}
        )
    )
    return vars
