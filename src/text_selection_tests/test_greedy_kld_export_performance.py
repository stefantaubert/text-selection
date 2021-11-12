from collections import OrderedDict

from ordered_set import OrderedSet
from text_selection.greedy_kld_export_performance import \
    greedy_kld_uniform_ngrams_seconds_with_preselection_perf


def test_sort_greedy_kld_until_with_preselection__without_preselection():
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
    seconds=target_duration,
    preselection_keys=OrderedSet(),
    select_from_durations_s=durations,
    select_from_keys=data.keys(),
    ignore_symbols={"b"},
    n_gram=1,
    duration_boundary=(0, 2),
    maxtasksperchild=None,
    n_jobs=1,
    batches=None,
    chunksize=1,
  )

  assert OrderedSet([5, 7]) == res
