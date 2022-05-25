from collections import OrderedDict

from ordered_set import OrderedSet

from text_selection_core.types import Dataset


def test_zero_lines__creates_set():
  result = Dataset(0, "test")
  assert result.line_count == 0
  assert result.subsets == OrderedDict((
    ("test", OrderedSet()),
  ))


def test_one_line_creates_set_with_zero():
  result = Dataset(1, "test")
  assert result.line_count == 1
  assert result.subsets == OrderedDict((
    ("test", OrderedSet((0,))),
  ))


def test_two_lines_creates_set_with_zero_and_one():
  result = Dataset(2, "test")
  assert result.line_count == 2
  assert result.subsets == OrderedDict((
    ("test", OrderedSet((0, 1))),
  ))
