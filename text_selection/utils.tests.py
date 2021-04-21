import pickle
import unittest

from text_selection.utils import *


class UnitTests(unittest.TestCase):

  def __init__(self, methodName: str) -> None:
    super().__init__(methodName)

  def test_find_unlike_sets_n_too_big_raises_ValueError(self):
    data = [{1, 2, 3}, {1, 2, 4}]
    with self.assertRaises(ValueError):
      find_unlike_sets(data, n=3, seed=1111)

  def test_find_unlike_sets_n_is_length_of_sample_set_list(self):
    data = [{1, 2, 3}, {1, 2, 4}, {1, 2}]

    selected_set_idxs = find_unlike_sets(data, n=3, seed=1111)
    self.assertEqual(selected_set_idxs, {0, 1, 2})

  # def test_find_unlike_sets(self):
  #   with open("/tmp/data.pkl", "rb") as f:
  #     data = pickle.load(f)

  #   selected_set_idxs = find_unlike_sets(data, n=2, seed=1111)
  #   self.assertEqual(2, len(set(selected_set_idxs)))

  def test_find_unlike_sets__same_sets__choose_different_idxs(self):
    data = [set(range(10)) for _ in range(10)]

    selected_set_idxs = find_unlike_sets(data, n=2, seed=1111)
    self.assertEqual(2, len(set(selected_set_idxs)))

  def test_find_empty_clusters__empty_cluster_index_equals_n(self):
    cluster_labels = np.array([0, 1, 2, 2, 1, 0, 1])
    n = 3

    empty_cluster_index = find_empty_clusters(cluster_labels, n)
    self.assertEqual(3, empty_cluster_index)

  def test_find_empty_clusters__empty_cluster_index_is_smaller_than_n(self):
    cluster_labels = np.array([0, 1, 2, 2, 1, 0, 1, 3])
    n = 10

    empty_cluster_index = find_empty_clusters(cluster_labels, n)
    self.assertEqual(4, empty_cluster_index)

  def test_vectorize_set(self):
    sample_set = {1, 4, 5}
    max_id = 6
    res = vectorize_set(sample_set, max_id)
    self.assertEqual(res, [0, 1, 0, 0, 1, 1, 0])

  def test_replace_chosen_indices_that_correspond_to_empty_clusters_with_first_unused_indices(self):
    chosen_indices = [3, 6, 1]
    unselected_indices = [0, 2, 4, 5]
    first_empty_cluster_index = 2
    replace_chosen_indices_that_correspond_to_empty_clusters_with_first_unused_indices(
      chosen_indices, unselected_indices, first_empty_cluster_index, 3)

    self.assertEqual(chosen_indices, [3, 6, 0])

  def test_replace_chosen_indices_that_do_not_belong_to_corresponding_cluster(self):
    cluster_labels = np.array([1, 2, 2, 1, 0, 0, 1])
    cluster_dists = np.array([[5, 1, 4], [4, 3, 1], [7, 8, 9], [1, 1, 8],
                              [5, 7, 9], [1.5, 1, 3], [2, 0.5, 2]])
    chosen_indices = [3, 6, 1]
    first_empty_cluster_index = 2
    replace_chosen_indices_that_do_not_belong_to_corresponding_cluster(
      cluster_labels, cluster_dists, chosen_indices, first_empty_cluster_index)

    self.assertEqual(chosen_indices, [5, 6, 1])

  def test_vectorize_all_sets(self):
    sample_set_list = [{1, 4, 5}, {1, 4, 6}]
    max_id = 6
    res = vectorize_all_sets(sample_set_list, max_id)

    self.assertTrue(np.array_equal(res, np.array([[0, 1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1]])))

  def test_get_max_entry(self):
    sample_set_list = [{1, 4, 5}, {1, 4, 6}]
    res = get_max_entry(sample_set_list)

    self.assertEqual(res, 6)

  def test_get_chosen_sets(self):
    sample_set_list = [{1, 2, 3}, {2, 3, 4}, {5, 6, 7}, {8, 9, 0}]
    chosen_indices = {3, 0}
    res = get_chosen_sets(sample_set_list, chosen_indices)
    self.assertEqual(res, [{1, 2, 3}, {8, 9, 0}])

  def test_get_first_percent_20percent(self):
    data = OrderedSet([1, 2, 3, 4, 5])

    res = get_first_percent(data, 20)

    self.assertEqual(OrderedSet([1]), res)

  def test_get_first_percent_50percent__rounds_up(self):
    data = OrderedSet([1, 2, 3, 4, 5, 6, 7])

    res = get_first_percent(data, 50)

    self.assertEqual(OrderedSet([1, 2, 3, 4]), res)

  def test_get_first_percent_100percent__adds_all(self):
    data = OrderedSet([1, 2, 3, 4, 5, 6, 7])

    res = get_first_percent(data, 100)

    self.assertEqual(OrderedSet([1, 2, 3, 4, 5, 6, 7]), res)

  def test_get_filtered_ngrams__returns_ordered_dict(self):
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

    self.assertTrue(isinstance(res, OrderedDict))

  def test_get_filtered_ngrams__order_is_retained(self):
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

    self.assertEqual(assert_res, res)

  def test_get_filtered_ngrams__one_grams__return_one_grams(self):
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

    self.assertEqual(assert_res, res)

  def test_get_filtered_ngrams__two_grams__return_two_grams(self):
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

    self.assertEqual(assert_res, res)

  def test_get_filtered_ngrams__three_grams__return_three_grams(self):
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

    self.assertEqual(assert_res, res)

  def test_get_filtered_ngrams__one_grams_filtered__return_filtered_one_grams(self):
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

    self.assertEqual(assert_res, res)

  def test_get_filtered_ngrams__two_grams_filtered__return_filtered_two_grams(self):
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

    self.assertEqual(assert_res, res)

  def test_get_filtered_ngrams__three_grams_filtered__return_filtered_three_grams(self):
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

    self.assertEqual(assert_res, res)

  def test_get_filtered_ngrams__non_existing_ignored__are_ignored(self):
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

    self.assertEqual(assert_res, res)

  def test_get_filtered_ngrams__empty_list__do_nothing(self):
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

    self.assertEqual(assert_res, res)

  # region get_n_divergent_seconds
  def test_get_right_start_index__expect_index_following_prev_vec(self):
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

    self.assertTrue(isinstance(res, int))
    self.assertEqual(4, res)

  def test_get_right_start_index__expect_index_following_prev_vec_although_step_length_is_not_reached(self):
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

    self.assertTrue(isinstance(res, int))
    self.assertEqual(4, res)

  def test_get_right_start_index__dur_sum_will_equal_step_length(self):
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

    self.assertTrue(isinstance(res, int))
    self.assertEqual(2, res)

  def test_get_right_start_index__dur_sum_will_get_bigger_than_step_length(self):
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

    self.assertTrue(isinstance(res, int))
    self.assertEqual(2, res)

  def test_get_right_start_index__durs_differ(self):
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

    self.assertTrue(isinstance(res, int))
    self.assertEqual(1, res)

  def test_get_n_divergent_seconds__one_iteration(self):
    n_data = 4
    durations = OrderedDict({k: 1 for k in range(n_data)})

    res = get_n_divergent_seconds(
      durations_s=durations,
      seconds=3,
      n=1,
    )

    self.assertEqual(1, len(res))
    self.assertEqual(3, len(res[0]))

  def test_get_n_divergent_seconds__two_iterations__no_overflowing(self):
    n_data = 2
    durations = OrderedDict({k: 1 for k in range(n_data)})

    res = get_n_divergent_seconds(
      durations_s=durations,
      seconds=1,
      n=2,
    )

    self.assertEqual(2, len(res))
    self.assertEqual(1, len(res[0]))
    self.assertEqual(1, len(res[1]))
    self.assertEqual(0, len(set(res[0]).intersection(set(res[1]))))

  def test_get_n_divergent_seconds__two_iterations__with_overflowing_once(self):
    n_data = 3
    durations = OrderedDict({k: 1 for k in range(n_data)})

    res = get_n_divergent_seconds(
      durations_s=durations,
      seconds=2,
      n=2,
    )

    self.assertEqual(2, len(res))
    self.assertEqual(1, len(set(res[0]).intersection(set(res[1]))))

  def test_get_n_divergent_seconds__two_iterations__with_overflowing_twice(self):
    n_data = 6
    durations = OrderedDict({k: 1 for k in range(n_data)})

    res = get_n_divergent_seconds(
      durations_s=durations,
      seconds=4,
      n=3,
    )

    self.assertEqual(3, len(res))
    self.assertEqual(2, len(set(res[0]).intersection(set(res[1]))))
    self.assertEqual(2, len(set(res[1]).intersection(set(res[2]))))
    self.assertEqual(2, len(set(res[0]).intersection(set(res[2]))))

  def test_get_n_divergent_seconds__with_different_durations(self):
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

    self.assertEqual(3, len(res))
    self.assertEqual(OrderedSet([0]), res[0])
    self.assertEqual(OrderedSet([1]), res[1])
    self.assertEqual(OrderedSet([2, 0]), res[2])

  def test_get_n_divergent_seconds__durations_differ__expect_three_times_all_keys_but_first_is_in_different_order(self):
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

    self.assertEqual(3, len(res))
    self.assertEqual(OrderedSet([0, 1, 2]), res[0])
    self.assertEqual(OrderedSet([1, 2, 0]), res[1])
    self.assertEqual(OrderedSet([1, 2, 0]), res[2])

  def test_get_n_divergent_seconds__same_durations__expect_three_times_all_keys_but_all_in_different_order(self):
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

    self.assertEqual(3, len(res))
    self.assertEqual(OrderedSet([0, 1, 2]), res[0])
    self.assertEqual(OrderedSet([1, 2, 0]), res[1])
    self.assertEqual(OrderedSet([2, 0, 1]), res[2])

  def test_get_n_divergent_seconds__with_many_different_durations(self):
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

    self.assertEqual(3, len(res))
    self.assertEqual(OrderedSet([0, 1, 2, 3]), res[0])
    self.assertEqual(OrderedSet([2, 3, 4, 5]), res[1])
    self.assertEqual(OrderedSet([5, 6, 0, 1]), res[2])

  # endregion
if __name__ == '__main__':
  suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
  unittest.TextTestRunner(verbosity=2).run(suite)
