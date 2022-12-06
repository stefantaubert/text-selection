
from collections import OrderedDict
from logging import getLogger

from ordered_set import OrderedSet

from text_selection_core.common import SelectionDefaultParameters
from text_selection_core.filtering.regex_filter import filter_regex_pattern
from text_selection_core.types import Dataset
from text_selection_core.validation import ValidationErr


def test_match_component():
  lines = ["x y z", "aa a xx", "bb b c", ""]
  ds = Dataset(len(lines), "base")
  sel_param = SelectionDefaultParameters(ds, OrderedSet(("base",)), "test")

  changed_anything = filter_regex_pattern(sel_param, lines, ".*([ab]+).*", "match", getLogger())

  assert isinstance(changed_anything, bool)
  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("base", OrderedSet((0, 3))),
    ("test", OrderedSet((1, 2))),
  ))


def test_find_component():
  lines = ["x y z", "aa a aa aa xx", "bb b c", ""]
  ds = Dataset(len(lines), "base")
  sel_param = SelectionDefaultParameters(ds, OrderedSet(("base",)), "test")

  changed_anything = filter_regex_pattern(sel_param, lines, "[ab]+", "find", getLogger())

  assert isinstance(changed_anything, bool)
  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("base", OrderedSet((0, 3))),
    ("test", OrderedSet((1, 2))),
  ))


def test_find_component_multiple_groups():
  lines = ["It as In memory o April twenty-first on it but us don't need th things to make we remember it tho we re wonderful glad t ave em from th doctor"]
  ds = Dataset(len(lines), "base")
  sel_param = SelectionDefaultParameters(ds, OrderedSet(("base",)), "test")

  changed_anything = filter_regex_pattern(
    sel_param, lines, "[^:] ([b-z])(([^a-zA-Z'])|($))", "find", getLogger())

  assert isinstance(changed_anything, bool)
  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("base", OrderedSet()),
    ("test", OrderedSet((0,))),
  ))


def test_component_no_group():
  lines = ["x y z", "a bb", "bb b c", ""]
  ds = Dataset(len(lines), "base")
  sel_param = SelectionDefaultParameters(ds, OrderedSet(("base",)), "test")

  changed_anything = filter_regex_pattern(sel_param, lines, ".*a.*", "match", getLogger())

  assert isinstance(changed_anything, bool)
  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("base", OrderedSet((0, 2, 3))),
    ("test", OrderedSet((1,))),
  ))


def test_invalid_regex():
  lines = ["x y z", "a bb", "bb b c", ""]
  ds = Dataset(len(lines), "base")
  sel_param = SelectionDefaultParameters(ds, OrderedSet(("base",)), "test")

  changed_anything = filter_regex_pattern(sel_param, lines, ".*a).*", "match", getLogger())

  assert isinstance(changed_anything, ValidationErr)
  assert changed_anything.default_message == "Regex pattern is invalid! Details: unbalanced parenthesis at position 3"
