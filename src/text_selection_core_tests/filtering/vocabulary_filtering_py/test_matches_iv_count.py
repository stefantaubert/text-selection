from text_selection_core.filtering.vocabulary_filtering import matches_iv_count


def test_empty_a_0__returns_true():
  assert matches_iv_count([], ["a"], 0)


def test_empty_a_1__returns_false():
  assert not matches_iv_count([], ["a"], 1)


def test_abb_b_2__returns_true():
  assert matches_iv_count(["a", "b", "b"], ["b"], 2)


def test_aba_c_2__returns_false():
  assert not matches_iv_count(["a", "b", "a"], ["c"], 2)
