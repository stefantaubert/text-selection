from collections import OrderedDict

from ordered_set import OrderedSet
from text_selection.greedy_kld_export_performance import \
    greedy_kld_uniform_ngrams_seconds_with_preselection_perf


def test_sort_greedy_kld_until_with_preselection__without_preselection():
  preselection = OrderedDict()
  data = OrderedDict([
    (5, ("a", "b")),
    (6, ("b",)),
    (7, ("c",)),
  ])

  durations = {
    5: 1.0,
    6: 1.0,
    7: 1.0,
  }

  target_duration = 2.0

  res = greedy_kld_uniform_ngrams_seconds_with_preselection_perf(
    data=data,
    durations_s=durations,
    seconds=target_duration,
    preselection=preselection,
    ignore_symbols={"b"},
    n_gram=1,
    chunksize=1,
    duration_boundary=(0, 2),
    maxtasksperchild=None,
    n_jobs=1,
  )

  assert OrderedSet([5, 7]) == res
