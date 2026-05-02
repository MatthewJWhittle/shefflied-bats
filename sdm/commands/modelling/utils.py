"Utility functions used across modelling scripts"

from typing import List

def clean_string(
    string: str,
    lowercase: bool = True
) -> str:
    """
    Convert a string to snake case.
    """
    if lowercase:
        string = string.lower()
    string = string.strip()
    string = string.replace(" ", "_")
    string = string.strip("_")
    return string

def get_model_id(
    parts : List[str]
) -> str:
    """
    Get a model identifier from a list of parts.
    """
    parts_cleaned = [clean_string(part) for part in parts]
    model_id = "_".join(parts_cleaned)
    return model_id

