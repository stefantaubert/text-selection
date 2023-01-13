from typing import OrderedDict

from ordered_set import OrderedSet

from text_selection.greedy.main import greedy_uniform_ngrams_seconds_with_preselection_perf


def test_component():
  data = OrderedDict([
    (4, ("d", "e")),
    (5, ("a", "b")),
    (6, ("b", "c", "a")),
    (7, ("c",)),
    (8, ("c", "c")),
    (9, ("c", "c")),
  ])

  preselection = OrderedSet([])
  select_from = OrderedSet([4, 5, 6, 7, 8, 9])

  select_from_durations = {
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 3.0,
  }

  target_duration = 4.0

  res = greedy_uniform_ngrams_seconds_with_preselection_perf(
    data=data,
    seconds=target_duration,
    preselection_keys=preselection,
    select_from_durations_s=select_from_durations,
    select_from_keys=select_from,
    ignore_symbols={"b"},
    n_gram=2,
    duration_boundary=(0, 2),
    maxtasksperchild=None,
    n_jobs=1,
    batches=None,
    chunksize=1,
  )

  assert res == OrderedSet((4, 6, 8, 5))
