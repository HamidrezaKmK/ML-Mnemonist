import os
from typing import Dict


def get_all_codes(parent_directory: str) -> Dict[str, str]:
    """
    Given a parent_directory it outputs a dictionary mapping
    the codes to their corresponding yaml file
    """
    all_codes = {}
    for f in os.listdir(parent_directory):
        if f.split('.')[-1] == 'yaml':
            all_codes[f.split('-MLM-')[0]] = os.path.join(parent_directory, f)

    return all_codes
