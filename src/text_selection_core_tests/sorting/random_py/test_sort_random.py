
from collections import OrderedDict
from logging import getLogger

from ordered_set import OrderedSet

from text_selection_core.common import SortingDefaultParameters
from text_selection_core.sorting.random import sort_random
from text_selection_core.types import Dataset


def test_0_0_a__returns_empty():
  ds = Dataset(0, "a")
  sel_param = SortingDefaultParameters(ds, OrderedSet(("a",)))

  changed_anything = sort_random(sel_param, seed=0, logger=getLogger())

  assert isinstance(changed_anything, bool)
  assert not changed_anything
  assert ds.subsets == OrderedDict((
    ("a", OrderedSet()),
  ))


def test_1_0_a__returns_0():
  ds = Dataset(1, "a")
  sel_param = SortingDefaultParameters(ds, OrderedSet(("a",)))

  changed_anything = sort_random(sel_param, seed=0, logger=getLogger())

  assert isinstance(changed_anything, bool)
  assert not changed_anything
  assert ds.subsets == OrderedDict((
    ("a", OrderedSet((0,))),
  ))


def test_2_1_a__returns_1_0():
  ds = Dataset(2, "a")
  sel_param = SortingDefaultParameters(ds, OrderedSet(("a",)))

  changed_anything = sort_random(sel_param, seed=1, logger=getLogger())

  assert isinstance(changed_anything, bool)
  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("a", OrderedSet((1, 0))),
  ))


def test_3_0_a__returns_0_2_1():
  ds = Dataset(3, "a")
  sel_param = SortingDefaultParameters(ds, OrderedSet(("a",)))

  changed_anything = sort_random(sel_param, seed=0, logger=getLogger())

  assert isinstance(changed_anything, bool)
  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("a", OrderedSet((0, 2, 1))),
  ))


def test_3_1_a__returns_1_2_0():
  ds = Dataset(3, "a")
  sel_param = SortingDefaultParameters(ds, OrderedSet(("a",)))

  changed_anything = sort_random(sel_param, seed=1, logger=getLogger())

  assert isinstance(changed_anything, bool)
  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("a", OrderedSet((1, 2, 0))),
  ))


def test_seed_is_subset_internal__3_1_ab__returns_1_2_0__4_5_3():
  ds = Dataset(3, "a")
  ds.subsets["b"] = OrderedSet((3, 4, 5))
  sel_param = SortingDefaultParameters(ds, OrderedSet(("a", "b")))

  changed_anything = sort_random(sel_param, seed=1, logger=getLogger())

  assert isinstance(changed_anything, bool)
  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("a", OrderedSet((1, 2, 0))),
    ("b", OrderedSet((4, 5, 3))),
  ))


def test_3_1_a_b__returns_1_2_0__3_4_5():
  ds = Dataset(3, "a")
  ds.subsets["b"] = OrderedSet((3, 4, 5))
  sel_param = SortingDefaultParameters(ds, OrderedSet(("a",)))

  changed_anything = sort_random(sel_param, seed=1, logger=getLogger())

  assert isinstance(changed_anything, bool)
  assert changed_anything
  assert ds.subsets == OrderedDict((
    ("a", OrderedSet((1, 2, 0))),
    ("b", OrderedSet((3, 4, 5))),
  ))
