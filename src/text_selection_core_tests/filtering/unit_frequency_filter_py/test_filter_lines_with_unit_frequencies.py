
from collections import OrderedDict
from logging import getLogger

from ordered_set import OrderedSet

from text_selection_core.common import SelectionDefaultParameters
from text_selection_core.filtering.unit_frequency_filter import (CountFilterParameters,
                                                                 filter_lines_with_unit_frequencies)
from text_selection_core.types import Dataset, move_lines_to_subset


def test_component():
  lines = [
    "a",
    "a b",
    "a b c",
    "a b c d",  # ...
    "a b c d e",  # 3x
    "a b c d e f",  # 2x
    "a b c d e f g",  # 1x
  ]
  ds = Dataset(len(lines), "base")
  move_lines_to_subset(ds, OrderedSet((0, 1)), "test", getLogger())

  params = CountFilterParameters(lines, " ", 1, 3, True, False, "any")

  changed_anything = filter_lines_with_unit_frequencies(
    SelectionDefaultParameters(ds, OrderedSet(("base",)), "selected"), params, getLogger())

  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("base", OrderedSet((2, 3, 4))),
    ("test", OrderedSet((0, 1))),
    ("selected", OrderedSet((5, 6))),
  ))


def test_component_percent():
  lines = [
    "a",
    "a b",
    "a b c",
    "a b c d",
    "a b c d e",
    "a b c d e f",
    "a b c d e f g",
  ]
  ds = Dataset(len(lines), "base")

  params = CountFilterParameters(lines, " ", 1, 63.54, True, True, "any")

  changed_anything = filter_lines_with_unit_frequencies(
    SelectionDefaultParameters(ds, OrderedSet(("base",)), "test"), params, getLogger())

  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("base", OrderedSet((0, 1, 2))),
    ("test", OrderedSet((3, 4, 5, 6))),
  ))
