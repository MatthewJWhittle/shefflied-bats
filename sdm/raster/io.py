from pathlib import Path
from typing import Dict, Any, Optional # Added Optional

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

def translate_to_cog(
    src_path: Path, 
    dst_path: Path, 
    profile: str = "webp", # common default for good compression/quality 
    profile_options: Optional[Dict[str, Any]] = None, # Changed to Optional[Dict]
    **options: Any # Pass other rio_cogeo options directly
) -> None:
    """Translates a raster to a Cloud Optimized GeoTIFF (COG).

    Args:
        src_path: Path to the source raster file.
        dst_path: Path to save the output COG file.
        profile: COG profile to use (e.g., "jpeg", "webp", "zstd", "lzw").
                 See rio-cogeo documentation for available profiles.
        profile_options: Dictionary of options for the chosen profile.
        **options: Additional keyword arguments to pass to cog_translate.
    """
    effective_profile_options = profile_options if profile_options is not None else {}

    # Get default profile options and update with any user-provided ones
    dst_profile = cog_profiles.get(profile)
    if not dst_profile:
        raise ValueError(f"Unknown COG profile: {profile}. Available: {list(cog_profiles.keys())}")
    
    # Create a mutable copy of the default profile to update
    final_dst_profile = dst_profile.copy()

    # Update default profile options with user-specified ones from effective_profile_options
    for key, value in effective_profile_options.items():
        if key in final_dst_profile:
            final_dst_profile[key] = value
        else:
            # Or, if you want to allow adding new keys to the profile dict (less safe)
            # final_dst_profile[key] = value 
            print(f"Warning: Option '{key}' not standard for profile '{profile}'. It will be passed to general cog_translate options.")
            # If an option is not part of the profile, pass it to general options
            if key not in options:
                 options[key] = value

    # Ensure output directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cog_translate(
        src_path,
        dst_path,
        final_dst_profile, # Use the merged profile dictionary
        **options,
    )
    # print(f"Successfully translated {src_path} to COG: {dst_path}") # Optional: add logging 