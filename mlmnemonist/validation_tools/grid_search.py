import copy
import json
import os
from typing import Optional, List, Dict

from mlmnemonist.runner_cache import RunnerCache

from mlmnemonist import ExperimentRunner
from mlmnemonist.validation_tools import expand_cfg
import functools

GRID_SEARCH_CACHING = 500


def _convert(code: str) -> List[int]:
    return [int(x) for x in code.split('-')]


def _rev_convert(code: List[int]) -> str:
    return '-'.join([str(x) for x in code])


def _get_all_codes(parent_directory: str) -> Dict[str, str]:
    all_codes = {}
    for f in os.listdir(parent_directory):
        if f.split('.')[-1] == 'yaml':
            all_codes[f.split('-MLM-')[0]] = os.path.join(parent_directory, f)

    return all_codes


def _get_adjacent_based_on_list(curr_list: List[int], guide: List[int], curr_index: int = 0) -> List[List[int]]:
    if curr_index == len(curr_list):
        return [copy.deepcopy(curr_list)]
    ret = []

    branch1 = copy.deepcopy(curr_list)
    branch1[curr_index] += 1
    if branch1[curr_index] == guide[curr_index]:
        branch1[curr_index] = 0
    if branch1[curr_index] != curr_list[curr_index]:
        ret += _get_adjacent_based_on_list(branch1, guide, curr_index + 1)

    branch2 = copy.deepcopy(curr_list)
    branch2[curr_index] -= 1
    if branch2[curr_index] == -1:
        branch2[curr_index] = guide[curr_index] - 1
    if branch2[curr_index] != curr_list[curr_index]:
        ret += _get_adjacent_based_on_list(branch2, guide, curr_index + 1)

    ret += _get_adjacent_based_on_list(curr_list, guide, curr_index + 1)

    return ret


@functools.lru_cache(maxsize=GRID_SEARCH_CACHING)
def _get_adjacent_codes(code: str, all_codes: str) -> List[str]:
    all_codes = all_codes.split('.')
    curr_list = [int(x) for x in code.split('-')]
    guide = []
    for code in all_codes:
        list_code = _convert(code)
        for i, x in enumerate(list_code):
            if i < len(guide):
                guide[i] = max(guide[i], x + 1)
            else:
                guide.append(x + 1)

    all_adj = _get_adjacent_based_on_list(curr_list, guide)
    return [_rev_convert(x) for x in all_adj]


def run_grid_search(runner: ExperimentRunner,
                    cache_token: str,
                    verbose: int,
                    save_directory: Optional[str] = None,
                    all_cfg_dir: Optional[str] = None,
                    cfg_dir: Optional[str] = None,
                    *args, **kwargs):
    # Create the config directories
    if cfg_dir is not None and save_directory is not None:
        cfg_dir = os.path.join(os.getenv('MLM_CONFIG_DIR'), cfg_dir)
        save_directory = os.path.join(os.getenv('MLM_EXPERIMENT_DIR'), save_directory)

        part1, part2 = os.path.split(save_directory)
        sv = os.getcwd()
        os.chdir(part1)
        if not os.path.exists(part2):
            os.mkdir(part2)
        os.chdir(sv)

        # Create a cache in save_directory
        cache = RunnerCache(directory=save_directory, token=cache_token)
        cache.LOAD()
        directories_done = cache.SET_IFN('directories_done', False)
        if not directories_done:
            expand_cfg(runner.cfg, cfg_dir, save_directory=save_directory)
        directories_done = True
        cache.SET('directories_done', directories_done)
        cache.SAVE()
        all_cfg_dir = save_directory
    else:
        all_cfg_dir = os.path.join(os.getenv('MLM_EXPERIMENT_DIR'), all_cfg_dir)
        cache = RunnerCache(directory=all_cfg_dir, token=cache_token)
        cache.LOAD()

    # Calculate the value for all iterations
    cfg_dict = _get_all_codes(all_cfg_dir)
    all_codes = list(cfg_dict.keys())
    curr = cache.SET_IFN('current_code', all_codes[0])
    marks = cache.SET_IFN('marks', set(list([curr])))
    score_dict = cache.SET_IFN('score_dict', {})

    last_score = None
    while True:
        found_more = False
        for adjacent_code in _get_adjacent_codes(curr, '.'.join(all_codes)):
            if adjacent_code not in marks:
                runner.verbose = verbose
                runner.CACHE.RESET(prompt=False)
                runner.cfg_path = cfg_dict[adjacent_code]
                print(f"Running {runner.cfg_path}")
                score = runner.run(*args, **kwargs)
                score_dict[runner.cfg_path] = score
                marks.add(adjacent_code)

                if last_score is None or score > last_score:
                    last_score = score
                    found_more = True
                    curr = adjacent_code
                    break

        cache.SET('current_code', curr)
        cache.SET('marks', marks)
        cache.SET('score_dict', score_dict)
        cache.SAVE()

        if not found_more:
            break

    with open(os.path.join(all_cfg_dir, 'seen_scores.json'), 'w') as f:
        json.dump(score_dict, f, indent=4)


if __name__ == '__main__':
    pass
