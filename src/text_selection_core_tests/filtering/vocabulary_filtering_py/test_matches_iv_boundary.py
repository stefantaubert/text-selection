from text_selection_core.filtering.vocabulary_filtering import matches_iv_boundary


def test_empty_a_0_1__returns_true():
  assert matches_iv_boundary([], ["a"], 0, 1)


def test_empty_a_1_2__returns_false():
  assert not matches_iv_boundary([], ["a"], 1, 2)


def test_ab_b_2_3__returns_false():
  assert not matches_iv_boundary(["a", "b"], ["b"], 2, 3)


def test_abb_b_2_3__returns_true():
  assert matches_iv_boundary(["a", "b", "b"], ["b"], 2, 3)


def test_abbb_b_2_3__returns_false():
  assert not matches_iv_boundary(["a", "b", "b", "b"], ["b"], 2, 3)
