import logging
from logging import getLogger

from ordered_set import OrderedSet

from text_selection.selection import SelectionMode
from text_selection_core.common import SelectionDefaultParameters, WeightSelectionParameters
from text_selection_core.selection.kld_selection import KldSelectionParameters, select_kld
from text_selection_core.types import Dataset, move_lines_to_subset
from text_selection_core_tests.selection.symbol_extractor_py.generate_test_data import \
  load_big_test_set


def xtest_stress_test_no_preselection():
  #lines = load_small_test_set()
  lines = load_big_test_set()
  # lines = lines[:10_000_000]  # 21.90235720493365s
  lines = lines[:160_000]

  dataset = Dataset(len(lines), "base")
  weights = [1] * len(lines)

  error, changed_anything = select_kld(SelectionDefaultParameters(dataset, OrderedSet(("base",)), "test"), KldSelectionParameters(
    lines, "|", False, SelectionMode.FIRST), WeightSelectionParameters(weights, 4, False, False), 10_000, 16, None, getLogger())


def xtest_stress_test_with_preselection():
  #lines = load_small_test_set()
  lines = load_big_test_set()
  # lines = lines[:10_000_000]  # 21.90235720493365s
  lines = lines[:160_000]

  dataset = Dataset(len(lines), "base")
  weights = [1] * len(lines)

  move_lines_to_subset(dataset, OrderedSet((1, 2, 8, 6, 13)), "test", getLogger())

  error, changed_anything = select_kld(SelectionDefaultParameters(dataset, OrderedSet(("base",)), "test"), KldSelectionParameters(
    lines, "|", True, SelectionMode.FIRST), WeightSelectionParameters(weights, 4, True, False), 10_000, 16, None, getLogger())


def xtest_stress_test_with_preselection_and_subset():
  #lines = load_small_test_set()
  lines = load_big_test_set()
  # lines = lines[:10_000_000]  # 21.90235720493365s
  lines = lines[:160_000]

  dataset = Dataset(len(lines), "base")
  move_lines_to_subset(dataset, OrderedSet((1, 2, 8, 6, 13)), "to", getLogger())
  move_lines_to_subset(dataset, OrderedSet((75, 32, 459, 5, 4, 123, 884)), "from", getLogger())

  error, changed_anything = select_kld(SelectionDefaultParameters(dataset, OrderedSet(("from",)), "to"), KldSelectionParameters(
    lines, "|", True, SelectionMode.FIRST), 30, 10_000, 16, None, getLogger())


if __name__ == "__main__":
  main_logger = getLogger()
  main_logger.setLevel(logging.DEBUG)
  main_logger.manager.disable = logging.NOTSET
  xtest_stress_test_no_preselection()
