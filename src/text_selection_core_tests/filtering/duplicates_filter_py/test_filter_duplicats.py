
from collections import OrderedDict
from logging import getLogger

from ordered_set import OrderedSet

from text_selection_core.common import SelectionDefaultParameters
from text_selection_core.filtering.duplicates_filter import filter_duplicates
from text_selection_core.types import Dataset


def test_component():
  ds = Dataset(3, "base")
  lines = ["a", "a", "b"]
  sel_param = SelectionDefaultParameters(ds, OrderedSet(("base",)), "test")

  changed_anything = filter_duplicates(sel_param, lines, None, getLogger())

  assert isinstance(changed_anything, bool)
  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("base", OrderedSet((0, 2))),
    ("test", OrderedSet((1,))),
  ))
