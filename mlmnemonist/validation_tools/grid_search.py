import copy
import json
import os
from typing import Optional, List, Dict

from mlmnemonist.runner_cache import RunnerCache
from mlmnemonist import ExperimentRunner
from mlmnemonist.validation_tools import expand_cfg
from yacs.config import CfgNode as ConfigurationNode

import functools

from mlmnemonist.validation_tools.utils import get_all_codes

GRID_SEARCH_CACHING = 500


def _convert(code: str) -> List[int]:
    return [int(x) for x in code.split('-')]


def _rev_convert(code: List[int]) -> str:
    return '-'.join([str(x) for x in code])




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
    """
    :param code:
    A code made up of digits separated by '-'

    :param all_codes:
    All the possible codes in our grid world

    :return:
    A list of codes adjacent to the current code
    """
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
    all_adj = [_rev_convert(x) for x in all_adj]
    all_adj = [x for x in all_adj if x in all_codes]
    return all_adj


def _run_grid_search(runner: ExperimentRunner,
                     cache: RunnerCache,
                     verbose: int,
                     cfg_base: ConfigurationNode,
                     all_cfg_dir: str,
                     with_preprocess: bool,
                     *args, **kwargs):
    cache.LOAD()
    # Add configurations
    runner.add_config(cfg_base=cfg_base)

    # Calculate the value for all iterations
    cfg_dict = get_all_codes(all_cfg_dir)
    all_codes = list(cfg_dict.keys())
    curr = cache.SET_IFN('current_code', all_codes[0])
    marks = cache.SET_IFN('marks', set(list([curr])))
    score_dict = cache.SET_IFN('score_dict', {})
    iteration_i = cache.SET_IFN('iteration_i', 0)

    last_score = None
    while True:
        found_more = False
        for adjacent_code in _get_adjacent_codes(curr, '.'.join(all_codes)):
            if adjacent_code not in marks:
                runner.verbose = max(0, verbose - 1)
                runner.cfg_path = cfg_dict[adjacent_code]

                if verbose > 0:
                    print(f"Iteration no. [{iteration_i + 1}/{len(all_codes)}] "
                          f"-- Running {os.path.split(runner.cfg_path)[-1]}", end=' : ')
                if with_preprocess:
                    runner.preprocess()
                score = runner.run(*args, **kwargs)
                runner.CACHE.RESET(prompt=False)
                if verbose > 0:
                    print(score)

                iteration_i += 1
                score_dict[runner.cfg_path] = score
                marks.add(adjacent_code)

                cache.SET('iteration_i', iteration_i)
                cache.SET('current_code', curr)
                cache.SET('marks', marks)
                cache.SET('score_dict', score_dict)
                cache.SAVE()

                with open(os.path.join(all_cfg_dir, 'seen_scores.json'), 'w') as f:
                    json.dump(score_dict, f, indent=4)

                if last_score is None or score > last_score:
                    last_score = score
                    found_more = True
                    curr = adjacent_code
                    break
        if not found_more:
            break

    if verbose > 0:
        print(f"Results saved in {os.path.join(all_cfg_dir, 'seen_scores.json')}!")
    return cache


def grid_search_from_palette(runner: ExperimentRunner,
                             cache_token: str,
                             verbose: int,
                             cfg_base: ConfigurationNode,
                             cfg_palette_dir: str,
                             save_directory: str,
                             with_preprocess: bool = False,
                             *args, **kwargs
                             ):
    cfg_palette_dir = os.path.join(os.getenv('MLM_CONFIG_DIR'), cfg_palette_dir)
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
        expand_cfg(cfg_base, cfg_palette_dir, save_directory=save_directory)
    directories_done = True
    cache.SET('directories_done', directories_done)
    cache.SAVE()
    all_cfg_dir = save_directory
    _run_grid_search(runner, cache, verbose, cfg_base, all_cfg_dir, with_preprocess, *args, **kwargs)


def grid_search(runner: ExperimentRunner,
                cache_token: str,
                verbose: int,
                cfg_base: ConfigurationNode,
                all_cfg_dir: str,
                with_preprocess: bool = False,
                *args, **kwargs):
    all_cfg_dir = os.path.join(os.getenv('MLM_EXPERIMENT_DIR'), all_cfg_dir)
    cache = RunnerCache(directory=all_cfg_dir, token=cache_token)
    cache.LOAD()
    _run_grid_search(runner, cache, verbose, cfg_base, all_cfg_dir, with_preprocess, *args, **kwargs)
