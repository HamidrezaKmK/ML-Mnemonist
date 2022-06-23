from collections import Callable
from typing import List, Optional, Union

Callable_Runner_With_Optional = Callable


class Pipeline:
    """
    This class contains a set of functions piped together.
    """

    def __init__(self):
        self._all_functions: List[Callable_Runner_With_Optional] = []

    def __str__(self):
        return [x.__name__ for x in self._all_functions].__str__()

    def _find_index_in_pipeline(self, func: Callable_Runner_With_Optional) -> Optional[int]:
        """
        Find the index of function func in the pipeline
        """
        for i in range(len(self._all_functions)):
            f = self._all_functions[i]
            if f.__name__ == func.__name__:
                return i
        return None

    @property
    def function_count(self):
        """
        :return:
        The number of functions in the pipeline
        """
        return len(self._all_functions)

    def update_function(self, func: Callable_Runner_With_Optional) -> None:
        """
        This function updates the func if available and if not available
        then simply appends the function in the preprocessing pipeline
        """
        i = self._find_index_in_pipeline(func)
        if i is None:
            self.add_function(func)
        else:
            self._all_functions[i] = func

    def insert_function(self, func: Callable_Runner_With_Optional, index: int) -> None:
        """
        insert func at index 'index' of the pipeline 
        """
        if self._find_index_in_pipeline(func) is not None:
            raise Exception("Function is repeated!")
        self._all_functions.insert(index, func)

    def add_function(self, func: Callable_Runner_With_Optional) -> None:
        if self._find_index_in_pipeline(func) is not None:
            raise Exception("Function is repeated")
        self._all_functions.append(func)

    def remove_function(self, func: Callable_Runner_With_Optional) -> None:
        """
        Removes the function from pipeline if it is currently available in the pipeline
        """
        if self._find_index_in_pipeline(func) is not None:
            self._all_functions.remove(func)

    def clear_functions(self) -> None:
        """
        Clear the preprocessing pipeline
        """
        self._all_functions.clear()

    def run(self, keep: bool, verbose: int, runner, *args, **kwargs) -> None:
        """
        Runs the whole pipeline. If the keep flag is set to true, then the pipeline would remain the same
        after running. However, if it set to false, then all the things that are ran will be removed.
        """
        begin = True
        nxt_layer = None
        for i, f in enumerate(self._all_functions):
            if verbose > 0:
                print(f"[{i + 1}/{len(self._all_functions)}] Running {f.__name__}")
            if begin:
                nxt_layer = f(runner, *args, **kwargs)
                begin = False
            else:
                if nxt_layer is not None:
                    nxt_layer = f(runner, nxt_layer)
                else:
                    nxt_layer = f(runner)
        if not keep:
            self.clear_functions()
        return nxt_layer
