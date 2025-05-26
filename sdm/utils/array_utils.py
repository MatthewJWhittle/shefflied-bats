import xarray as xr

def calculate_z_score(
        array: xr.DataArray,
        eps: float = 1e-8,
) -> xr.DataArray:
    """
    Calculate the z-score of an array.
    Args:
        array (xr.DataArray): The input array.
        eps (float): Epsilon to avoid division by zero if std is zero.
    Returns:
        xr.DataArray: The z-score of the input array.
    """
    mean = array.mean(skipna=True)
    std = array.std(skipna=True)
    # Ensure std is not zero or too close to zero
    std_safe = xr.where(std < eps, eps, std)
    z_score = (array - mean) / std_safe
    return z_score 