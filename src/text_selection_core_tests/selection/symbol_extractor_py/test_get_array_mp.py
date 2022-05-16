import logging
import random
import string
from functools import partial
from itertools import chain
from logging import getLogger
from multiprocessing import Pool, cpu_count
from time import perf_counter

import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_core.selection.symbol_extractor import get_array, get_array_mp
from text_selection_core.types import Lines
from text_selection_core_tests.selection.symbol_extractor_py.generate_test_data import (
  load_big_test_set, load_small_test_set)


def test_stress_test():
  #lines = load_small_test_set()
  lines = load_big_test_set()
  lines = lines[:10_000_000]  # 21.90235720493365s
  lines = lines[:5_000_000]  # 21.90235720493365s
  #lines = lines[:160_000]
  start = perf_counter()
  array, symbols = get_array_mp(lines, range(len(lines)), "", getLogger(), 5_00000, 16, None)
  duration = perf_counter() - start
  main_logger.info(f"Duration: {duration}s")


if __name__ == "__main__":
  main_logger = getLogger()
  main_logger.setLevel(logging.DEBUG)
  main_logger.manager.disable = logging.NOTSET
  test_stress_test()
