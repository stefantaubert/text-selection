import logging
import random
import string
from functools import partial
from itertools import chain
from logging import getLogger
from multiprocessing import Pool, cpu_count

from ordered_set import OrderedSet
from tqdm import tqdm

from text_selection_core.selection.symbol_extractor import get_array
from text_selection_core.types import Lines
from text_selection_core_tests.selection.symbol_extractor_py.generate_test_data import (
  load_big_test_set, load_small_test_set)


def test_stress_test():
  #lines = load_small_test_set()
  lines = load_big_test_set()
  #lines = lines[:10_000_000]
  #lines = lines[:100_000]
  get_array(lines, range(len(lines)), "|", getLogger())


if __name__ == "__main__":
  main_logger = getLogger()
  main_logger.setLevel(logging.DEBUG)
  main_logger.manager.disable = logging.NOTSET
  test_stress_test()
