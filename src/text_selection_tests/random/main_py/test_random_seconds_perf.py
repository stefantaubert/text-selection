from ordered_set import OrderedSet

from text_selection.random.main import random_seconds_perf


def test_component():
  select_from = OrderedSet([5, 6, 7, 8, 9])

  select_from_durations = {
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 3.0,
  }

  target_duration = 4.0

  res = random_seconds_perf(
    seconds=target_duration,
    select_from_durations_s=select_from_durations,
    select_from_keys=select_from,
    duration_boundary=(0, 2),
    seed=1234,
  )

  assert res == OrderedSet((8, 5, 6, 7))
