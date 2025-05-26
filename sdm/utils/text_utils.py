import re

def tidy_variable_name(name: str) -> str:
    """
    Cleans up a string to be a suitable variable name by:
    - Replacing dashes, spaces, and other common separators with underscores.
    - Converting to lowercase.
    - Stripping leading/trailing whitespace and underscores.
    - Ensuring no multiple consecutive underscores.

    Args:
        name (str): The input string.

    Returns:
        str: The cleaned up string, suitable for use as a variable name.
    """
    if not isinstance(name, str):
        try:
            name = str(name) # Attempt to convert to string if not already
        except Exception:
            # Or raise a TypeError if strict string input is required
            raise TypeError(f"Input name must be a string or convertible to a string, got {type(name)}")

    # Replace common separators and problematic characters with underscores
    name = re.sub(r'[\s\-/\\.:;,()\[\]{}]', '_', name)
    # Convert to lowercase
    name = name.lower()
    # Remove any characters not alphanumeric or underscore
    name = re.sub(r'[^a-z0-9_]', '', name)
    # Replace multiple underscores with a single underscore
    name = re.sub(r'_+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    
    # Ensure it doesn't start with a number if it's meant for some contexts (e.g. Python var names)
    # For raster band names, this might be less critical.
    # if name and name[0].isdigit():
    #     name = f"_{name}" # Prepend underscore
        
    return name 