import logging
from logging import getLogger
from math import ceil
from time import perf_counter

from text_selection_core.selection.symbol_extractor import get_array_mp
from text_selection_core_tests.selection.symbol_extractor_py.generate_test_data import \
  load_big_test_set


def test_stress_test():
  #lines = load_small_test_set()
  lines = load_big_test_set()
  select = 10_000_000
  select = 5_000_000
  lines = lines[:select]  # 21.90235720493365s
  # lines = lines[:select]  # 21.90235720493365s
  #lines = lines[:160_000]
  start = perf_counter()
  array, symbols = get_array_mp(lines, range(len(lines)), "", {"", "a", "b"},
                                getLogger(), ceil(select / 16), 16, None)
  duration = perf_counter() - start
  main_logger.info(f"Duration: {duration}s")


if __name__ == "__main__":
  main_logger = getLogger()
  main_logger.setLevel(logging.DEBUG)
  main_logger.manager.disable = logging.NOTSET
  test_stress_test()
