import os
from pyhere import here

def set_project_wd(verbose=True):
    # Navigate to your project directory and create a '.here' file if it doesn't exist
    project_dir = here(".")
    os.chdir(project_dir)

    # Verify that the working directory has been changed
    if verbose:
        print("Current Working Directory:", os.getcwd())
    
    return None
