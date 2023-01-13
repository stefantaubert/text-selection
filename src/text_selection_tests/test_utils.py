import pytest

from text_selection.utils import *


def test_get_distribution():
  data = {
    2: [1, 2, 3],
    3: [1, 2],
    4: [1],
  }

  x = get_distribution(data)

  assert_res = {
    1: 3 / 6,
    2: 2 / 6,
    3: 1 / 6,
  }

  assert assert_res == x


def test_get_reverse_distribution():
  data = {
    2: [1, 2, 3],
    3: [1, 2],
    4: [1],
  }

  x = get_reverse_distribution(data)

  assert_res = {
    1: 1 / 6,
    2: 2 / 6,
    3: 3 / 6,
  }

  assert assert_res == x


def test_get_common_durations():
  chosen_sets = [{0, 1}, {1, 2}, {0, 1, 2}]
  durations_s = {0: 1, 1: 2, 2: 2.5}
  res = get_common_durations(chosen_sets, durations_s)

  assert len(res) == 3
  assert res[(0, 1)] == 2
  assert res[(0, 2)] == 3
  assert res[(1, 2)] == 4.5


def test_get_chosen_sets():
  sample_set_list = [{1, 2, 3}, {2, 3, 4}, {5, 6, 7}, {8, 9, 0}]
  chosen_indices = {3, 0}
  res = get_chosen_sets(sample_set_list, chosen_indices)
  assert res == [{1, 2, 3}, {8, 9, 0}]


def test_get_first_percent_20percent():
  data = OrderedSet([1, 2, 3, 4, 5])

  res = get_first_percent(data, 20)

  assert OrderedSet([1]) == res


def test_get_first_percent_50percent__rounds_up():
  data = OrderedSet([1, 2, 3, 4, 5, 6, 7])

  res = get_first_percent(data, 50)

  assert OrderedSet([1, 2, 3, 4]) == res


def test_get_first_percent_100percent__adds_all():
  data = OrderedSet([1, 2, 3, 4, 5, 6, 7])

  res = get_first_percent(data, 100)

  assert OrderedSet([1, 2, 3, 4, 5, 6, 7]) == res


# def test_get_filtered_ngrams_numbers__returns_ordered_dict():
#   data = OrderedDict([
#     (1, ["a", "b"]),
#     (3, ["e", "f"]),
#     (2, ["c", "d"]),
#     (5, ["i", "j"]),
#     (4, ["g", "a"]),
#   ])

#   res = get_filtered_ngrams_numbers(
#     data=data,
#     ignore_symbols=None,
#     n_gram=1,
#   )

#   assert_result = OrderedDict([
#     (1, (0, 1)),
#     (3, (2, 3)),
#     (2, (4, 5)),
#     (5, (6, 7)),
#     (4, (8, 0)),
#   ])

#   assert res == assert_result


def test_get_filtered_ngrams__returns_ordered_dict():
  data = OrderedDict({
    1: ["a", "b"],
    3: ["e", "f"],
    2: ["c", "d"],
    5: ["i", "j"],
    4: ["g", "h"],
  })

  res = get_filtered_ngrams(
    data=data,
    ignore_symbols=None,
    n_gram=1,
  )

  assert isinstance(res, OrderedDict)


def test_get_filtered_ngrams__order_is_retained():
  data = OrderedDict({
    1: ["a", "b"],
    3: ["e", "f"],
    2: ["c", "d"],
    5: ["i", "j"],
    4: ["g", "h"],
  })

  res = get_filtered_ngrams(
    data=data,
    ignore_symbols=None,
    n_gram=1,
  )

  assert_res = OrderedDict({
    1: [("a",), ("b",)],
    3: [("e",), ("f",)],
    2: [("c",), ("d",)],
    5: [("i",), ("j",)],
    4: [("g",), ("h",)],
  })

  assert assert_res == res


def test_get_filtered_ngrams__one_grams__return_one_grams():
  data = OrderedDict({
    1: ["a", "b"],
    2: ["c", "a"],
  })

  res = get_filtered_ngrams(
    data=data,
    ignore_symbols=None,
    n_gram=1,
  )

  assert_res = OrderedDict({
    1: [("a",), ("b",)],
    2: [("c",), ("a",)],
  })

  assert assert_res == res


def test_get_filtered_ngrams__two_grams__return_two_grams():
  data = OrderedDict({
    1: ["a", "b"],
    2: ["c", "a"],
  })

  res = get_filtered_ngrams(
    data=data,
    ignore_symbols=None,
    n_gram=2,
  )

  assert_res = OrderedDict({
    1: [("a", "b",)],
    2: [("c", "a",)],
  })

  assert assert_res == res


def test_get_filtered_ngrams__three_grams__return_three_grams():
  data = OrderedDict({
    1: ["a", "b", "x"],
    2: ["c", "a", "y"],
  })

  res = get_filtered_ngrams(
    data=data,
    ignore_symbols=None,
    n_gram=3,
  )

  assert_res = OrderedDict({
    1: [("a", "b", "x")],
    2: [("c", "a", "y")],
  })

  assert assert_res == res


def test_get_filtered_ngrams__one_grams_filtered__return_filtered_one_grams():
  data = OrderedDict({
    1: ["a", "b"],
    2: ["c", "a"],
  })

  res = get_filtered_ngrams(
    data=data,
    ignore_symbols={"a"},
    n_gram=1,
  )

  assert_res = OrderedDict({
    1: [("b",)],
    2: [("c",)],
  })

  assert assert_res == res


def test_get_filtered_ngrams__two_grams_filtered__return_filtered_two_grams():
  data = OrderedDict({
    1: ["a", "b"],
    2: ["c", "a"],
  })

  res = get_filtered_ngrams(
    data=data,
    ignore_symbols={"b"},
    n_gram=2,
  )

  assert_res = OrderedDict({
    1: [],
    2: [("c", "a",)],
  })

  assert assert_res == res


def test_get_filtered_ngrams__three_grams_filtered__return_filtered_three_grams():
  data = OrderedDict({
    1: ["a", "b", "x"],
    2: ["c", "a", "y"],
  })

  res = get_filtered_ngrams(
    data=data,
    ignore_symbols={"x"},
    n_gram=3,
  )

  assert_res = OrderedDict({
    1: [],
    2: [("c", "a", "y")],
  })

  assert assert_res == res


def test_get_filtered_ngrams__non_existing_ignored__are_ignored():
  data = OrderedDict({
    1: ["a", "b"],
    2: ["c", "a"],
  })

  res = get_filtered_ngrams(
    data=data,
    ignore_symbols={"x"},
    n_gram=1,
  )

  assert_res = OrderedDict({
    1: [("a",), ("b",)],
    2: [("c",), ("a",)],
  })

  assert assert_res == res


def test_get_filtered_ngrams__empty_list__do_nothing():
  data = OrderedDict({
    1: [],
    2: [],
  })

  res = get_filtered_ngrams(
    data=data,
    ignore_symbols=None,
    n_gram=1,
  )

  assert_res = OrderedDict({
    1: [],
    2: [],
  })

  assert assert_res == res

# region get_n_divergent_seconds


def test_get_right_start_index__expect_index_following_prev_vec():
  n_data = 8
  step_length = 4
  durations = OrderedDict({index: 1 for index in range(n_data)})
  prev_vec = [0, 1, 2, 3]
  data_keys = list(range(n_data))

  res = get_next_start_index(
    step_length=step_length,
    durations_s=durations,
    prev_vec=prev_vec,
    data_keys=data_keys
  )

  assert isinstance(res, int)
  assert 4 == res


def test_get_right_start_index__expect_index_following_prev_vec_although_step_length_is_not_reached():
  n_data = 8
  step_length = 5
  durations = OrderedDict({index: 1 for index in range(n_data)})
  prev_vec = [0, 1, 2, 3]
  data_keys = list(range(n_data))

  res = get_next_start_index(
    step_length=step_length,
    durations_s=durations,
    prev_vec=prev_vec,
    data_keys=data_keys
  )

  assert isinstance(res, int)
  assert 4 == res


def test_get_right_start_index__dur_sum_will_equal_step_length():
  n_data = 8
  step_length = 4
  durations = OrderedDict({index: 2 for index in range(n_data)})
  prev_vec = [0, 1, 2, 3]
  data_keys = list(range(n_data))

  res = get_next_start_index(
    step_length=step_length,
    durations_s=durations,
    prev_vec=prev_vec,
    data_keys=data_keys
  )

  assert isinstance(res, int)
  assert 2 == res


def test_get_right_start_index__dur_sum_will_get_bigger_than_step_length():
  n_data = 8
  step_length = 3
  durations = OrderedDict({index: 2 for index in range(n_data)})
  prev_vec = [0, 1, 2, 3]
  data_keys = list(range(n_data))

  res = get_next_start_index(
    step_length=step_length,
    durations_s=durations,
    prev_vec=prev_vec,
    data_keys=data_keys
  )

  assert isinstance(res, int)
  assert 2 == res


def test_get_right_start_index__durs_differ():
  step_length = 2
  durations = OrderedDict({
    0: 1,
    1: 7,
    2: 1
  })
  prev_vec = [0, 1, 2]
  data_keys = list(range(3))

  res = get_next_start_index(
    step_length=step_length,
    durations_s=durations,
    prev_vec=prev_vec,
    data_keys=data_keys
  )

  assert isinstance(res, int)
  assert 1 == res


def test_get_n_divergent_seconds__should_end_in_error():
  durations = OrderedDict({
    0: 1,
    1: 7,
    2: 1
  })
  seconds = 3
  n = 3
  with pytest.raises(AssertionError):
    get_n_divergent_seconds(durations, seconds, n)


def test_get_n_divergent_seconds__one_iteration():
  n_data = 4
  durations = OrderedDict({k: 1 for k in range(n_data)})

  res = get_n_divergent_seconds(
    durations_s=durations,
    seconds=3,
    n=1,
  )

  assert 1 == len(res)
  assert 3 == len(res[0])


def test_get_n_divergent_seconds__two_iterations__no_overflowing():
  n_data = 2
  durations = OrderedDict({k: 1 for k in range(n_data)})

  res = get_n_divergent_seconds(
    durations_s=durations,
    seconds=1,
    n=2,
  )

  assert 2 == len(res)
  assert 1 == len(res[0])
  assert 1 == len(res[1])
  assert 0 == len(set(res[0]).intersection(set(res[1])))


def test_get_n_divergent_seconds__two_iterations__with_overflowing_once():
  n_data = 3
  durations = OrderedDict({k: 1 for k in range(n_data)})

  res = get_n_divergent_seconds(
    durations_s=durations,
    seconds=2,
    n=2,
  )

  assert 2 == len(res)
  assert 1 == len(set(res[0]).intersection(set(res[1])))


def test_get_n_divergent_seconds__two_iterations__with_overflowing_twice():
  n_data = 6
  durations = OrderedDict({k: 1 for k in range(n_data)})

  res = get_n_divergent_seconds(
    durations_s=durations,
    seconds=4,
    n=3,
  )

  assert 3 == len(res)
  assert 2 == len(set(res[0]).intersection(set(res[1])))
  assert 2 == len(set(res[1]).intersection(set(res[2])))
  assert 2 == len(set(res[0]).intersection(set(res[2])))


def test_get_n_divergent_seconds__with_different_durations():
  durations = OrderedDict({
    0: 1,
    1: 7,
    2: 1
  })
  res = get_n_divergent_seconds(
    durations_s=durations,
    seconds=7,
    n=3
  )

  assert 3 == len(res)
  assert OrderedSet([0]) == res[0]
  assert OrderedSet([1]) == res[1]
  assert OrderedSet([2, 0]) == res[2]


def test_get_n_divergent_seconds__durations_differ__expect_three_times_all_keys_but_first_is_in_different_order():
  durations = OrderedDict({
    0: 1,
    1: 7,
    2: 1
  })
  res = get_n_divergent_seconds(
    durations_s=durations,
    seconds=9,
    n=3
  )

  assert 3 == len(res)
  assert OrderedSet([0, 1, 2]) == res[0]
  assert OrderedSet([1, 2, 0]) == res[1]
  assert OrderedSet([1, 2, 0]) == res[2]


def test_get_n_divergent_seconds__same_durations__expect_three_times_all_keys_but_all_in_different_order():
  durations = OrderedDict({
    0: 1,
    1: 1,
    2: 1
  })
  res = get_n_divergent_seconds(
    durations_s=durations,
    seconds=3,
    n=3
  )

  assert 3 == len(res)
  assert OrderedSet([0, 1, 2]) == res[0]
  assert OrderedSet([1, 2, 0]) == res[1]
  assert OrderedSet([2, 0, 1]) == res[2]


def test_get_n_divergent_seconds__with_many_different_durations():
  durations = OrderedDict({
    0: 1,
    1: 2,
    2: 3,
    3: 1,
    4: 1,
    5: 2,
    6: 2
  })

  res = get_n_divergent_seconds(
    durations_s=durations,
    seconds=7,
    n=3,
  )

  assert 3 == len(res)
  assert OrderedSet([0, 1, 2, 3]) == res[0]
  assert OrderedSet([2, 3, 4, 5]) == res[1]
  assert OrderedSet([5, 6, 0, 1]) == res[2]

# endregion


def test_filter_after_duration__includes_from_excludes_to():
  durations = {
    0: 1,
    1: 2,
    2: 3,
    3: 2.9999,
    4: 1,
    5: 2,
  }

  result = filter_after_duration(
    corpus=durations,
    max_duration_excl=3,
    min_duration_incl=2,
  )

  assert result == {1, 3, 5}
