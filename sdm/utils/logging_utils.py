import logging
import sys

def setup_logging(level=logging.INFO, verbose: bool = False):
    """Basic logging configuration."""
    log_level = logging.DEBUG if verbose else level
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    # Quiet down some overly verbose loggers if necessary
    # logging.getLogger("rasterio").setLevel(logging.WARNING)
    # logging.getLogger("fiona").setLevel(logging.WARNING) 