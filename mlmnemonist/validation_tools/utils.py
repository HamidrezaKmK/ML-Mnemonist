import os
from typing import Dict


def get_all_codes(parent_directory: str) -> Dict[str, str]:
    """
    This function is called on a parent directory that contains all the
    configuration files. It returns a dictionary of all the codes and their
    corresponding configuration file paths.

    For example:
    > parent_directory
        0-0-MLM-conf-0.yaml
        1-0-MLM-conf-0.yaml
        0-1-MLM-conf-0.yaml
    
    Furthermore, the function returns a dictionary of the form:
    {
        "0-0": "0-0-MLM-conf-0.yaml",
        "1-0": "1-0-MLM-conf-0.yaml",
        "0-1": "0-1-MLM-conf-0.yaml"
    }
    """
    all_codes = {}
    for f in os.listdir(parent_directory):
        if f.split('.')[-1] == 'yaml':
            all_codes[f.split('-MLM-')[0]] = os.path.join(parent_directory, f)

    return all_codes
