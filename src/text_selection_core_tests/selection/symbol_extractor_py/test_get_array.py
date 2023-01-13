import logging
from logging import getLogger
from time import perf_counter

from text_selection_core.selection.symbol_extractor import get_array
from text_selection_core_tests.selection.symbol_extractor_py.generate_test_data import \
  load_big_test_set


def xtest_stress_test():
  #lines = load_small_test_set()
  lines = load_big_test_set()
  select = 10_000_000  # 21.90235720493365s
  select = 1_000_000
  select = 5_000_000  # 51.598537511890754s
  lines = lines[:select]
  start = perf_counter()
  array, symbols = get_array(lines, range(len(lines)), "", {"a", "b", ""})
  duration = perf_counter() - start
  print(f"Duration: {duration}s")
  main_logger = getLogger()
  main_logger.info(f"Duration: {duration}s")


if __name__ == "__main__":
  main_logger = getLogger()
  main_logger.setLevel(logging.DEBUG)
  main_logger.manager.disable = logging.NOTSET
  xtest_stress_test()
