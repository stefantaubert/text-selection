from text_selection_core.filtering.vocabulary_filtering import matches_any_iv


def test_empty__returns_false():
  assert not matches_any_iv([], ["a"])


def test_ab_b__returns_true():
  assert matches_any_iv(["a", "b"], ["b"])


def test_ab_c__returns_false():
  assert not matches_any_iv(["a", "b"], ["c"])
