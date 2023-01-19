from text_selection_core.filtering.vocabulary_filtering import matches_oov_count


def test_empty_0__returns_true():
  assert matches_oov_count([], ["a"], 0)


def test_empty_1__returns_false():
  assert not matches_oov_count([], ["a"], 1)


def test_abb_a_2__returns_true():
  assert matches_oov_count(["a", "b", "b"], ["a"], 2)


def test_abb_b_2__returns_false():
  assert not matches_oov_count(["a", "b", "b"], ["b"], 2)


def test_abc_b_2__returns_true():
  assert matches_oov_count(["a", "b", "c"], ["b"], 2)
