import logging
import random
from logging import getLogger
from time import perf_counter

import numpy as np

from text_selection_core.selection.symbol_extractor import get_array
from text_selection_core_tests.selection.symbol_extractor_py.generate_test_data import (
  load_big_test_set)


def test_stress_test():
  #lines = load_small_test_set()
  lines = load_big_test_set()
  # lines = lines[:10_000_000]  # 21.90235720493365s
  lines = lines[:160_000]
  array, symbols = get_array(lines, range(len(lines)), "", getLogger())
  indices = random.choices(range(10_000_000), k=9_000_000)
  indices = np.array(indices)
  main_logger = getLogger()
  start = perf_counter()
  result = array[indices, :]
  duration = perf_counter() - start
  main_logger.info(f"Duration: {duration}s")


if __name__ == "__main__":
  main_logger = getLogger()
  main_logger.setLevel(logging.DEBUG)
  main_logger.manager.disable = logging.NOTSET
  test_stress_test()
