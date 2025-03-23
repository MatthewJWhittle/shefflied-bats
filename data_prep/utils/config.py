import logging

def setup_logging(log_level=logging.INFO, log_format=None):
    """
    Set up logging configuration.
    
    Parameters:
    log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    log_format (str): Custom log format. If None, a default format will be used.
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(level=log_level, format=log_format)

